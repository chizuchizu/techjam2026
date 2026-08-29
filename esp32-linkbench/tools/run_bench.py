#!/usr/bin/env python3
"""run_bench.py - flash + host-drive the esp32-linkfast 2-node ESP-NOW benchmark.

Usage:
  python3 run_bench.py <mode> [--env-server ENV] [--env-client ENV] [--flash] [--timeout S]

Modes: 'opt' (default) or 'compat'.
"""
import argparse, json, os, re, subprocess, sys, time
import serial

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_SERVER='/dev/cu.usbmodem101'
P_CLIENT='/dev/cu.usbmodem1101'
ENVS={'opt':('link-server','link-client-ctl'),'compat':('link-server-compat','link-client-compat')}

PING_DONE_RE=re.compile(r"PING\|done\|P=(\d+)\|n=(\d+)\|lost=(\d+)\|rtt_med=(\d+)\|rtt_min=(\d+)\|rtt_max=(\d+)\|rtt_p95=(\d+)\|jit=(\d+)\|srv_med=(\d+)\|cs_med=(\d+)\|sc_med=(\d+)")
CLIENT_RE=re.compile(r"CLIENT\|P=(\d+)\|N=(\d+)\|sent=(\d+)\|fail=(\d+)\|err=(\d+)\|acked=(\d+)\|bytes=(\d+)\|us=(\d+)\|thr=(\d+)\|rtt_us_med=(\d+)")
RX_RE=re.compile(r"SERVER\|rx\|pkts=(\d+)\|bytes=(\d+)")
BOOT_RE=re.compile(r"LINKFW\|role=(SERVER|CLIENT)\|mac=([0-9A-F:]+)\|ch=(\d+)\|mode=(\w+)")

def flash(env, port):
    r=subprocess.run(['pio','run','-e',env,'-t','upload','--upload-port',port],
                     cwd=ROOT, capture_output=True, text=True, timeout=180)
    return 'SUCCESS' in r.stdout, r.stdout[-300:]

def parse(text):
    out={'ping':[],'stream':[],'server_rx':None,'boot':{}}
    for raw in text.splitlines():
        line=raw.strip()
        m=BOOT_RE.search(line)
        if m: out['boot'][m.group(1)]={'mac':m.group(2),'ch':int(m.group(3)),'mode':m.group(4)}; continue
        m=PING_DONE_RE.search(line)
        if m:
            g=[int(x) for x in m.groups()]
            out['ping'].append(dict(zip(['P','n','lost','rtt_med','rtt_min','rtt_max','rtt_p95','jit','srv_med','cs_med','sc_med'],g)))
            continue
        m=CLIENT_RE.search(line)
        if m:
            g=[int(x) for x in m.groups()]
            out['stream'].append(dict(zip(['P','N','sent','fail','err','acked','bytes','us','thr','rtt_us_med'],g)))
            continue
        m=RX_RE.search(line)
        if m: out['server_rx']={'pkts':int(m.group(1)),'bytes':int(m.group(2))}
    return out

ESPPORT = '/Users/karthikgangula/.platformio/packages/tool-esptoolpy/esptool.py'
def force_boot(port):
    """read_mac ends with a hard reset -> reliably exits ROM download into the app."""
    r=subprocess.run(['python3',ESPPORT,'--chip','esp32c3','-p',port,'--baud','115200','read_mac'],
                     capture_output=True, text=True, timeout=30)
    return r.returncode==0

def safe_read(s, var):
    try:
        n=s.in_waiting
        if n:
            return s.read(n).decode(errors='replace')
    except serial.SerialException:
        pass
    return ''

def run(mode, do_flash, timeout):
    env_s, env_c=ENVS[mode]
    if do_flash:
        print(f'[1/3] flashing server {env_s}...', flush=True)
        ok,err=flash(env_s,P_SERVER)
        print('   OK' if ok else '   FAIL '+err[-200:], flush=True)
        print(f'[2/3] flashing client {env_c}...', flush=True)
        ok,err=flash(env_c,P_CLIENT)
        print('   OK' if ok else '   FAIL '+err[-200:], flush=True)
    # after flash the chip can sit in ROM download (macOS C3 quirk); force a clean
    # boot on both, then the client reaches READY ~2-4 s after (join + probe).
    print('[3/3] forcing boot + opening serial + driving bench...', flush=True)
    force_boot(P_SERVER); time.sleep(1.0)
    force_boot(P_CLIENT)
    ss=serial.Serial(P_SERVER,115200,timeout=0.3)
    sc=serial.Serial(P_CLIENT,115200,timeout=0.3)
    time.sleep(0.3)
    try:
        bs=''; bc=''; t0=time.time(); sent_b=False; ready=False
        ready_wait=0
        while time.time()-t0<timeout:
            bs+=safe_read(ss,'bs'); bc+=safe_read(sc,'bc')
            if not ready and 'CLIENT|READY' in bc:
                ready=True; ready_wait=time.time()
                print('   client ready -> sending B', flush=True)
                sc.write(b'B'); sc.flush(); sent_b=True
                t0=time.time()  # restart clock: give the bench its own window
            # fallback: client reprints READY every 2s; if we somehow miss it but
            # the board is clearly up, send B after a fixed settle anyway
            if not sent_b and 'CLIENT|boot' in bc and time.time()-t0>3:
                print('   board booted, no READY seen -> sending B', flush=True)
                sc.write(b'B'); sc.flush(); sent_b=True; t0=time.time()
            if sent_b and 'CLIENT|DONE' in bc:
                time.sleep(1.5)
                bs+=safe_read(ss,'bs'); bc+=safe_read(sc,'bc')
                break
            time.sleep(0.03)
    finally:
        ss.close(); sc.close()
    d=os.path.join(ROOT,'.logs'); os.makedirs(d,exist_ok=True)
    open(os.path.join(d,f'server_{mode}.log'),'w').write(bs)
    open(os.path.join(d,f'client_{mode}.log'),'w').write(bc)
    res=parse(bc); res['server_boot']=parse(bs)['boot'].get('SERVER')
    res['server_rx']=res['server_rx'] or parse(bs)['server_rx']
    jp=os.path.join(d,f'result_{mode}.json')
    ok_capture = res['ping'] or res['stream']
    if not ok_capture:
        print('!! capture produced NO ping/stream events — keeping previous result, NOT overwriting', jp)
        return res
    json.dump(res,open(jp,'w'),indent=1)
    print(f"\n==== {mode.upper()} ====  (server boot={res['server_boot']})")
    print('PING latency (us): P    n  lost   med   min   max   p95   jit  srv_med  cs_med  sc_med')
    for x in res['ping']:
        print('  %4d %5d %5d %5d %5d %5d %5d %5d %7d %6d %6d'%(
            x['P'],x['n'],x['lost'],x['rtt_med'],x['rtt_min'],x['rtt_max'],x['rtt_p95'],x['jit'],x['srv_med'],x['cs_med'],x['sc_med']))
    print('STREAM:   P   N sent fail err acked  thr(B/s)  rtt_med(us)')
    for x in res['stream']:
        print('  %4d %4d %4d %4d %3d %5d %9d %10d'%(x['P'],x['N'],x['sent'],x['fail'],x['err'],x['acked'],x['thr'],x['rtt_us_med']))
    if res['server_rx']:
        print('server ground-truth rx:', res['server_rx'])
    print('saved', jp)
    return res

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode', nargs='?', default='opt')
    ap.add_argument('--env-server'); ap.add_argument('--env-client')
    ap.add_argument('--no-flash', action='store_true')
    ap.add_argument('--timeout', type=int, default=200)
    a=ap.parse_args()
    if a.env_server and a.env_client:
        ENVS[a.mode]=(a.env_server,a.env_client)
    run(a.mode, do_flash=not a.no_flash, timeout=a.timeout)
