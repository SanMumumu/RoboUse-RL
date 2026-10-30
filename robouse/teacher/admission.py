"""Read-only identity/quota rules retained from the existing Astra collector."""
import base64
import hashlib
import json
import math
from pathlib import Path
import time
from agp_rl.io import read_json, sha256

AUTH_NS = "https://api.openai.com/auth"


class FrontierBlocked(RuntimeError):
    pass


def hashed(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def jwt_claims(value):
    try:
        part = value.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        if not isinstance(claims, dict):
            raise ValueError()
        return claims
    except (ValueError, IndexError, AttributeError):
        raise FrontierBlocked("LOGIN_CLAIMS_UNAVAILABLE") from None


def login_identity(source, *, minimum_lifetime=90):
    """Read only the chosen file. Return hashes/times; never return any token."""
    try:
        data = read_json(Path(source) / "auth.json")
        tokens = data["tokens"]
        identity, access = (
            jwt_claims(tokens["id_token"]),
            jwt_claims(tokens["access_token"]),
        )
        account = tokens["account_id"]
        user = identity.get(AUTH_NS, {}).get("chatgpt_user_id") or identity.get("sub")
        email = identity.get("email")
        expires = access["exp"]
        if (
            data.get("auth_mode") not in (None, "chatgpt")
            or data.get("OPENAI_API_KEY")
            or (not account)
            or (not user)
            or (not email)
        ):
            raise FrontierBlocked("NATIVE_CHATGPT_LOGIN_REQUIRED")
        for claims in (identity, access):
            claim_account = claims.get(AUTH_NS, {}).get("chatgpt_account_id")
            if claim_account and claim_account != account:
                raise FrontierBlocked("LOGIN_IDENTITY_INCONSISTENT")
        if (
            type(expires) not in (int, float)
            or not math.isfinite(expires)
            or expires <= time.time() + minimum_lifetime
        ):
            raise FrontierBlocked("LOGIN_EXPIRED_OR_REQUIRES_REFRESH")
        return dict(
            account_sha256=hashed(account),
            user_sha256=hashed(user),
            email_sha256=hashed(email),
            access_expires_at=expires,
            credential_file_sha256=sha256(Path(source) / "auth.json"),
        )
    except (KeyError, OSError, ValueError, TypeError):
        raise FrontierBlocked("LOGIN_METADATA_UNAVAILABLE") from None


def normalize_quota(raw, identity, observed_at):
    """Unknown/expired/credits are closed; usedPercent is not remainingPercent."""
    if not isinstance(raw, dict):
        raise FrontierBlocked("QUOTA_RESPONSE_INVALID")
    overall = raw.get("rateLimits")
    if not isinstance(overall, dict):
        raise FrontierBlocked("QUOTA_OVERALL_UNKNOWN")
    extra_buckets = raw.get("rateLimitsByLimitId") or {}
    if not isinstance(extra_buckets, dict) or "overall" in extra_buckets:
        raise FrontierBlocked("QUOTA_BUCKETS_INVALID")
    buckets = {"overall": overall, **extra_buckets}
    windows, reasons = ([], [])
    for name, bucket in buckets.items():
        if not isinstance(bucket, dict):
            raise FrontierBlocked("QUOTA_BUCKET_INVALID")
        credits = bucket.get("credits") or {}
        if not isinstance(credits, dict):
            raise FrontierBlocked("QUOTA_CREDITS_INVALID")
        if credits.get("unlimited") or (
            credits.get("hasCredits") and raw.get("ordinaryUsageAllowed") is not True
        ):
            reasons.append("CREDIT_FALLBACK_NOT_AUTHORIZED")
        if bucket.get("rateLimitReachedType") is not None:
            reasons.append("PROVIDER_RATE_LIMIT_REACHED")
        count = 0
        for kind in ("primary", "secondary"):
            w = bucket.get(kind)
            if w is None:
                continue
            try:
                used, minutes, reset = (
                    w["usedPercent"],
                    w["windowDurationMins"],
                    w["resetsAt"],
                )
                if any(
                    (
                        type(x) not in (int, float) or not math.isfinite(x)
                        for x in (used, minutes, reset)
                    )
                ):
                    raise ValueError()
                if not 0 <= used <= 100 or minutes <= 0 or reset <= observed_at:
                    raise ValueError()
            except (KeyError, TypeError, ValueError):
                raise FrontierBlocked("QUOTA_WINDOW_INVALID") from None
            windows.append(
                dict(
                    bucket=name,
                    window=kind,
                    used_percent=used,
                    remaining_percent=100 - used,
                    minutes=minutes,
                    resets_at=reset,
                )
            )
            count += 1
        if not count:
            raise FrontierBlocked("QUOTA_WINDOW_UNKNOWN")
    if not any((w["bucket"] == "overall" and w["minutes"] == 10080 for w in windows)):
        raise FrontierBlocked("QUOTA_WEEKLY_UNKNOWN")
    if raw.get("ordinaryUsageAllowed") is False:
        reasons.append("INCLUDED_USAGE_UNAVAILABLE")
    remaining = min((w["remaining_percent"] for w in windows))
    if remaining <= 20:
        reasons.append("REMAINING_NOT_ABOVE_20_PERCENT")
    return dict(
        account_sha256=identity["account_sha256"],
        user_sha256=identity["user_sha256"],
        observed_at=observed_at,
        remaining_percent=remaining,
        windows=windows,
        allowed=not reasons,
        reasons=reasons,
        source="account/rateLimits/read",
        inference_requests=0,
        quota_delta_is_not_run_token_usage=True,
    )
