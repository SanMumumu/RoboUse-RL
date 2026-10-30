"""Credential-free RPC classification; an internal code is not a VPN diagnosis."""
import re

DENIED=('LOGIN','PERMISSION','_QUOTA','PROVIDER_RATE_LIMIT_REACHED','WORKSPACE_REQUIREMENTS','LOCAL_IO',
        'IDENTITY_MISMATCH','SETTINGS','CREDENTIAL','STORAGE','READ_ONLY','WRITABLE_NFS')


def rpc_fault(method,error,base_classification):
    error=error if isinstance(error,dict) else {}
    data=error.get('data') if isinstance(error.get('data'),dict) else {}
    status=next((x for x in [error.get('status'),error.get('statusCode'),
        data.get('status'),data.get('statusCode'),data.get('http_status')]
        if type(x)==int and x in (401,403,429)),None)
    # Inspect but never retain arbitrary provider text or identifiers.
    if status is None:
        match=re.search(r'(?<!\d)(401|403|429)(?!\d)',str(error.get('message','')))
        if match:status=int(match.group())
    classification=base_classification
    if status:classification=f'METADATA_{method.replace("/","_")}_HTTP_{status}'
    return dict(classification=classification,rpc_method=method,
        rpc_code=error.get('code') if type(error.get('code'))==int else None,http_status=status)


def retry_kind(exc):
    text=str(exc).upper()
    if any(word in text for word in DENIED) or any(f'HTTP_{x}' in text for x in (401,403,429)):
        return 'HOLD'
    if getattr(exc,'http_status',None) in (401,403,429):return 'HOLD'
    if getattr(exc,'rpc_code',None)==-32603 and getattr(exc,'rpc_method',None) in ('account/read','model/list'):
        if text.endswith(('_UNKNOWN','_INTERNAL')):return 'INTERNAL_CANDIDATE'
    if any(word in text for word in ('NETWORK','METADATA_TIMEOUT','TRANSPORT_FAILURE',
                                    'CONNECTION RESET','CONNECTION REFUSED','PROXY')):
        return 'NETWORK'
    return 'HOLD'
