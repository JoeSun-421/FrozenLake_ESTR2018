// End-to-end checks using the installed browser and Node's built-in CDP support.
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import net from 'node:net';
import assert from 'node:assert/strict';

const root = path.resolve(import.meta.dirname, '..');
async function freePort() { const server = net.createServer(); server.listen(0, '127.0.0.1'); await once(server, 'listening'); const port = server.address().port; await new Promise(r => server.close(r)); return port; }
const delay = ms => new Promise(r => setTimeout(r, ms));
async function until(fn, label) { for (let i=0;i<100;i++) { try { if(await fn())return; } catch {} await delay(100); } throw Error('Timed out: '+label); }
const port = await freePort(), debug = await freePort();
const profile = await mkdtemp(path.join(tmpdir(), 'frozenlake-viewer-'));
const server = spawn(path.join(root, '.venv/bin/python'), ['-m','frozenlake','view','--port',String(port)], {cwd:root,stdio:'ignore'});
let browser, socket;
try {
  await until(async() => (await fetch(`http://127.0.0.1:${port}/gui/`)).ok, 'viewer server');
  browser = spawn(process.env.CHROME || '/usr/bin/google-chrome', ['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',`--remote-debugging-port=${debug}`,`--user-data-dir=${profile}`,'--no-first-run','about:blank'], {stdio:'ignore'});
  await until(async()=> (await fetch(`http://127.0.0.1:${debug}/json/version`)).ok, 'browser');
  const tab = await (await fetch(`http://127.0.0.1:${debug}/json/new?${encodeURIComponent(`http://127.0.0.1:${port}/gui/`)}`, {method:'PUT'})).json();
  socket = new WebSocket(tab.webSocketDebuggerUrl); await once(socket, 'open');
  let id=0;const pending=new Map(),errors=[];
  socket.addEventListener('message',event=>{const msg=JSON.parse(event.data);if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params);if(msg.id){const p=pending.get(msg.id);pending.delete(msg.id);if(msg.error)p.reject(Error(JSON.stringify(msg.error)));else p.resolve(msg.result);}});
  const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});socket.send(JSON.stringify({id:key,method,params}));});
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.text);return r.result.value;};
  await send('Runtime.enable'); await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1280,height:1100,deviceScaleFactor:1,mobile:false});
  const catalog=JSON.parse(await readFile(path.join(root,'results/catalog.json'),'utf8'));
  for(let i=0;i<catalog.experiments.length;i++) {
    const entry=catalog.experiments[i];
    if(i)await evaluate(`document.getElementById('experiment').value='${i}';document.getElementById('experiment').dispatchEvent(new Event('change'));`);
    await until(async()=>await evaluate(`document.body?.dataset.loaded===${JSON.stringify(entry.id)}`),entry.id);
    const cfg=JSON.parse(await readFile(path.join(root,entry.path,'config.json'),'utf8'));
    assert.equal(await evaluate(`document.body.dataset.checkpoint`),String(cfg.training.interactions));
    await evaluate(`document.getElementById('step').value='0';document.getElementById('step').dispatchEvent(new Event('input'));`);
    assert.equal(await evaluate('document.body.dataset.checkpoint'),'0');
    assert.equal(await evaluate(`document.querySelector('#values tbody tr td:nth-child(2)').textContent`),'0.000000');
    await evaluate(`document.getElementById('step').value=document.getElementById('step').max;document.getElementById('step').dispatchEvent(new Event('input'));`);
  }
  await evaluate(`document.getElementById('experiment').value='0';document.getElementById('experiment').dispatchEvent(new Event('change'));`);
  await until(async()=>await evaluate(`document.body.dataset.loaded===${JSON.stringify(catalog.experiments[0].id)}`),'reset first dataset');
  const original=await evaluate(`document.getElementById('map').toDataURL()`);
  await evaluate(`document.getElementById('mode').value='3';document.getElementById('mode').dispatchEvent(new Event('change'));`);
  assert.notEqual(await evaluate(`document.getElementById('map').toDataURL()`),original);
  await evaluate(`document.getElementById('mode').value='max';document.getElementById('mode').dispatchEvent(new Event('change'));`);
  await evaluate(`document.getElementById('map').scrollIntoView({block:'center'});`);
  const point=await evaluate(`(()=>{const r=document.getElementById('map').getBoundingClientRect();return{x:r.left+r.width*4.5/8,y:r.top+r.height*3.5/8};})()`);
  await send('Input.dispatchMouseEvent',{type:'mousePressed',...point,button:'left',buttons:1,clickCount:1});
  await send('Input.dispatchMouseEvent',{type:'mouseReleased',...point,button:'left',buttons:0,clickCount:1});
  await until(async()=>/\(4, 5\)/.test(await evaluate(`document.getElementById('cell').textContent`)),'cell selection');
  assert.match(await evaluate(`document.getElementById('cell').textContent`),/\(4, 5\)/);
  assert.equal(errors.length,0);
  const screenshot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
  const target=path.join(tmpdir(),'estr2018-viewer.png');await writeFile(target,Buffer.from(screenshot.data,'base64'));
  console.log('Viewer passed: all datasets, zero/final checkpoints, action-value modes and cell selection. Screenshot: '+target);
} finally {
  if(socket)socket.close();if(browser)browser.kill();server.kill();await delay(500);await rm(profile,{recursive:true,force:true});
}
