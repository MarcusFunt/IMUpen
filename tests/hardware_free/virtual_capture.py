#!/usr/bin/env python3
"""Deterministic synthetic IMUpen CSV producer (not a real IMU emulator).

Generates report-compatible host recordings and known injected failures. The
six-axis motion is physically consistent for an ideal fixed-position rotation
about Z, then biased, noised and quantized using LSM6DS3 nominal sensitivities.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import measure

RATE = 208.0
ACC_G_PER_LSB = 0.000061
GYRO_DPS_PER_LSB = 0.0175

def to_int16(v):
    return max(-32768, min(32767, round(v)))

def axes_for_time(t, rnd, accel_noise_mg, gyro_noise_dps, gyro_bias_dps, accel_bias_mg):
    # Physical orientation -> gravity in rotating IMU coordinates. The gyro
    # convention is chosen to agree with the sign in alignment_metrics.
    theta = 0.36 * math.sin(2*math.pi*0.43*t) + 0.13 * math.sin(2*math.pi*0.81*t)
    omega_z = -(0.36*2*math.pi*0.43*math.cos(2*math.pi*0.43*t) +
                0.13*2*math.pi*0.81*math.cos(2*math.pi*0.81*t))
    gauss = rnd.gauss
    accel_g = [math.cos(theta), math.sin(theta), 0.0]
    gyro_dps = [0.,0.,omega_z*180/math.pi]
    a = [to_int16((v+accel_bias_mg/1000 + gauss(0,accel_noise_mg/1000))/ACC_G_PER_LSB)
         for v in accel_g]
    g = [to_int16((v+gyro_bias_dps + gauss(0,gyro_noise_dps))/GYRO_DPS_PER_LSB)
         for v in gyro_dps]
    return g+a

def generate(output, seconds=10, scenario="clean", seed=123, accel_noise_mg=1.0,
             gyro_noise_dps=0.08, gyro_bias_dps=0.15, accel_bias_mg=5.0,
             odr_ppm=0.0, irq_jitter_us=60.0):
    """Output a synthetic *processed CSV*, not raw register or serial framing.

    For actual .ino register-level tests use native_harness.cpp.
    """
    output=Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rnd=random.Random(seed)
    batches=max(4,int(seconds*RATE)//8)
    period_us=1_000_000/RATE*(1+odr_ppm*1e-6)
    missing=[]
    truth={"scenario":scenario,"synthetic":True,"nominal_hz":RATE,"true_hz":1_000_000/period_us,
           "seed":seed,"batches":batches,"true_missing_sequence_numbers":0,
           "clipped_injected_frames":0,"epochs":1,"sessions":1,
           "limitations":["This is simulated processed output, not physical IMU data",
                          "USB, radio, actual FreeRTOS scheduling and electrical effects are not modeled"]}
    current_boot=111111
    session=epoch=seq=queue_drops=recoveries=overruns=0
    physical_frame=8 # simulate first eight incoming hardware frames after configuration
    with output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["host_elapsed_s","session"]+measure.FIELDS)
        for b in range(batches):
            if scenario=="reboot" and b==batches//2:
                session+=1;truth["sessions"]+=1
                current_boot+=1;seq=0;epoch=0;queue_drops=recoveries=overruns=0
                physical_frame+=16
            if scenario=="fifo_reset" and b==batches//2:
                epoch+=1;truth["epochs"]+=1;recoveries+=1
                physical_frame+=16
            irq_count=b+1 if session==0 else b-batches//2+1
            burst_start=physical_frame
            anchor_index=burst_start+7
            jitter=rnd.gauss(0,irq_jitter_us)
            if scenario=="irq_outliers" and b>0 and b%11==0:
                jitter+=25000
            irq_us=round(1_000_000+anchor_index*period_us+jitter)
            read_start_us=irq_us+40
            read_end_us=read_start_us+220
            for k in range(8):
                i=physical_frame
                physical_frame+=1
                t=(i*period_us)/1_000_000
                gx,gy,gz,ax,ay,az=axes_for_time(t,rnd,accel_noise_mg,gyro_noise_dps,
                                                gyro_bias_dps,accel_bias_mg)
                if scenario=="clipping" and i%29==0:
                    ax=32767
                    truth["clipped_injected_frames"]+=1
                if scenario in ("queue_drop","reboot") and seq>0 and seq%41==0:
                    queue_drops+=1;truth["true_missing_sequence_numbers"]+=1
                    missing.append({"session":session,"seq":seq})
                    seq+=1
                    continue
                pattern=2 if scenario=="nonzero_pattern" else 0
                word_end=(3+6*k) if pattern==2 else (5+6*k)
                host_time=(anchor_index*period_us+1500+40*k)/1_000_000
                values=[current_boot,0x69,seq,epoch,b,irq_count,irq_us,
                        read_start_us,read_end_us,48,pattern,word_end,
                        gx,gy,gz,ax,ay,az,queue_drops,recoveries,overruns]
                writer.writerow([f"{host_time:.6f}",session]+values)
                seq+=1
    truth["known_missing_rows"]=missing[:100]
    (output.parent / (output.stem+".truth.json")).write_text(json.dumps(truth,indent=2)+"\n",encoding="utf-8")
    return truth

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--seconds",type=float,default=10)
    p.add_argument("--scenario",choices=["clean","queue_drop","reboot","fifo_reset","irq_outliers","nonzero_pattern","clipping"],default="clean")
    p.add_argument("--seed",type=int,default=123)
    p.add_argument("--accel-noise-mg",type=float,default=1.0)
    p.add_argument("--gyro-noise-dps",type=float,default=0.08)
    p.add_argument("--gyro-bias-dps",type=float,default=0.15)
    p.add_argument("--accel-bias-mg",type=float,default=5.0)
    p.add_argument("--odr-ppm",type=float,default=0.0)
    p.add_argument("--irq-jitter-us",type=float,default=60.0)
    args=p.parse_args()
    if args.seconds<=0 or args.irq_jitter_us<0: p.error("seconds must be positive and jitter nonnegative")
    truth=generate(**vars(args))
    print(json.dumps({k:v for k,v in truth.items() if k!='known_missing_rows'},indent=2))

if __name__=="__main__":main()
