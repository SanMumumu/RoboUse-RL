// Trusted runner. All model code executes inside the enclosing empty-root bwrap.
'use strict';
const readline = require('readline');
const vm = require('vm');
const fs = require('fs');
const pending = new Map();
let sequence = 0;
const send = value => process.stdout.write(JSON.stringify(value) + '\n');
if (process.argv[2] === '--probe') {
  const paths = JSON.parse(process.argv[3]);
  send({denied: paths.map(path => {
    try { fs.accessSync(path, fs.constants.R_OK); return false; }
    catch (_) { return true; }
  })});
  process.exit(0);
}
let started = false;
readline.createInterface({input: process.stdin}).on('line', async line => {
  let msg;
  try { msg = JSON.parse(line); } catch (_) { process.exit(2); }
  if (msg.kind === 'reply') {
    const p = pending.get(msg.id);
    if (!p) return process.exit(2);
    pending.delete(msg.id);
    if (msg.error) p.reject(new Error(msg.error)); else p.resolve(msg.value);
    return;
  }
  if (started || msg.kind !== 'execute') return process.exit(2);
  started = true;
  const state = msg.state || {};
  const output = [];
  const tools = {};
  for (const name of msg.tools) {
    tools['mcp__robot_js__' + name] = args => new Promise((resolve, reject) => {
      const id = ++sequence;
      pending.set(id, {resolve, reject});
      send({kind: 'tool', id, name, arguments: args || {}});
    });
    tools['mcp__robot__' + name] = tools['mcp__robot_js__' + name];
  }
  const copy = value => JSON.parse(JSON.stringify(value));
  const context = vm.createContext({tools: Object.freeze(tools),
    text: value => output.push({type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value)}),
    image: value => {
      if (!value || value.type !== 'image' || typeof value.data !== 'string')
        throw new Error('image requires an actual MCP image block');
      output.push(copy(value));
    },
    store: (key, value) => { state[String(key)] = copy(value); },
    load: key => state[String(key)] === undefined ? undefined : copy(state[String(key)]),
  });
  try {
    await new vm.Script('(async () => {\n' + msg.code + '\n})()').runInContext(context, {timeout: msg.timeout_ms});
    if (pending.size) throw new Error('Unawaited tool operations are refused');
    send({kind: 'complete', state, output});
  } catch (error) {
    send({kind: 'complete', state, output, error: String(error.message).slice(0, 500)});
  }
  process.exit(0);
});
