"""Credential-free JSON-RPC subprocess fixture, never an Astra endpoint."""
import json,sys,time
from pathlib import Path

counter=Path(sys.argv[1]);stage=sys.argv[2];http=int(sys.argv[3])
for line in sys.stdin:
    request=json.loads(line)
    if 'id' not in request:continue
    method=request['method'];result={}
    if method==stage:
        count=int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count+1))
        if count<2 or http:
            error=dict(code=-32603,message=sys.argv[4] if len(sys.argv)>4 else 'workspace operation failed')
            if http:error['data']=dict(status=http)
            print(json.dumps(dict(id=request['id'],error=error)),flush=True);continue
    if method=='account/read':result=dict(account=dict(type='chatgpt',email='cpu-fixture@invalid'))
    elif method=='model/list':result=dict(data=[dict(model='gpt-6-astra',supportedReasoningEfforts=['high'])])
    elif method=='account/rateLimits/read':result=dict(rateLimits=dict(primary=dict(usedPercent=1,windowDurationMins=10080,resetsAt=time.time()+86400)))
    print(json.dumps(dict(id=request['id'],result=result)),flush=True)
