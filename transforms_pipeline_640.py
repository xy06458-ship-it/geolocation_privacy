#!/usr/bin/env python3
"""对照实验(640): 高斯模糊 + 黑色像素遮挡。分割用BiSeNetV2(Cityscapes 19类)"""
import os, sys, csv, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms, models
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
import clip
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXTERNAL_METHODS_DIR = PROJECT_ROOT / 'external_methods'
BISENET_ROOT = EXTERNAL_METHODS_DIR / 'bisenetv2'
DATA_ROOT = PROJECT_ROOT / 'data' / 'cityscapes'

sys.path.insert(0, str(EXTERNAL_METHODS_DIR))
sys.path.insert(0, str(BISENET_ROOT))

BISENET_WEIGHT = BISENET_ROOT / 'weights' / 'model_final_v2_city.pth'
BISENET_CONFIG = BISENET_ROOT / 'configs' / 'bisenetv2_city.py'
CLASSIFIER_DIR = DATA_ROOT / 'city_classifier'
OUTPUT_ROOT    = DATA_ROOT / '640' / 'results_transforms'
ORIG_CSV = DATA_ROOT / 'test_list.csv'
CSV_640  = DATA_ROOT / '640' / 'test_list_640.csv'

ATTACK_SIZE = (640, 640)
DEVICE = torch.device('cuda')
NUM_CLASSES = 14
START_IDX = 0
MAX_IMAGES = 1000
DENSITY_TOP_RATIO = 0.05
DENSITY_TOP_K = 3
STATIC_MASK_K = -0.3

# 黑色遮挡填充色
FILL = {'black':(0,0,0)}

EXPERIMENTS = []
for region,gm in [('full',None),('static',None),('resnet_grad','resnet'),('clip_grad','clip')]:
    EXPERIMENTS.append(dict(name=f'blur_{region}_s7', mode='blur', region=region, sigma=7, gmodel=gm, fill=None))
for region,gm in [('full',None),('static',None),('resnet_grad','resnet'),('clip_grad','clip')]:
    EXPERIMENTS.append(dict(name=f'mask_black_{region}', mode='mask', region=region, sigma=None, gmodel=gm, fill='black'))

from sensitivity.table import WEIGHTS as _W, CLASS_NAMES
STATIC_W = torch.tensor(_W, dtype=torch.float32)
CITYSCAPES_CLASSES = ['road','sidewalk','building','wall','fence','pole','traffic light',
   'traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']

seg_tf = transforms.Compose([transforms.Resize((512,1024)), transforms.ToTensor(),
    transforms.Normalize([0.3257,0.3690,0.3223],[0.2112,0.2148,0.2115])])
resnet_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
clip_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711])])

class CLIPFinetune(nn.Module):
    def __init__(self, cm, nc):
        super().__init__(); self.encoder = cm.visual; self.fc = nn.Linear(512, nc)
    def forward(self, x): return self.fc(self.encoder(x))

print("Loading BiSeNetV2 (Cityscapes)...")
from configs import set_cfg_from_file
from lib.models import model_factory
bisenet_cfg = set_cfg_from_file(str(BISENET_CONFIG))
bisenet_model = model_factory[bisenet_cfg.model_type](bisenet_cfg.n_cats, aux_mode='eval')
bisenet_checkpoint = torch.load(BISENET_WEIGHT, map_location='cpu')
if isinstance(bisenet_checkpoint, dict):
    for checkpoint_key in ('state_dict', 'model_state_dict', 'model'):
        if checkpoint_key in bisenet_checkpoint and isinstance(bisenet_checkpoint[checkpoint_key], dict):
            bisenet_checkpoint = bisenet_checkpoint[checkpoint_key]
            break
bisenet_state_dict = {
    key.replace('module.', '', 1) if key.startswith('module.') else key: value
    for key, value in bisenet_checkpoint.items()
    if torch.is_tensor(value)
}
bisenet_model.load_state_dict(bisenet_state_dict, strict=True)
bisenet_model.eval().to(DEVICE)

print("Loading classifiers...")
resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(2048, NUM_CLASSES)
resnet.load_state_dict(torch.load(os.path.join(CLASSIFIER_DIR,'resnet50_imagenet_best.pth'), map_location='cpu'))
resnet.eval().to(DEVICE)
base_clip,_ = clip.load('ViT-B/16', device='cpu'); base_clip = base_clip.float()
clip_ft = CLIPFinetune(base_clip, NUM_CLASSES)
clip_ft.load_state_dict(torch.load(os.path.join(CLASSIFIER_DIR,'clip_finetune_best.pth'), map_location='cpu'))
clip_ft.eval().to(DEVICE)
print("All models loaded.\n")

fid_tf = transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor(),
    transforms.Lambda(lambda x:(x*255).byte())])
lpips_fn = fid_metric = kid_metric = None
def load_eval_models():
    global lpips_fn, fid_metric, kid_metric
    if lpips_fn is not None: return
    import lpips as lpips_lib
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    print("Loading LPIPS/FID/KID...")
    lpips_fn = lpips_lib.LPIPS(net='alex').to(DEVICE)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
    kid_metric = KernelInceptionDistance(feature=2048, subset_size=50, normalize=False).to(DEVICE)
    print("Eval models loaded.")

@torch.no_grad()
def get_segmentation(img_pil):
    inp = seg_tf(img_pil.convert('RGB')).unsqueeze(0).to(DEVICE)
    out = bisenet_model(inp)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits = F.interpolate(logits, size=ATTACK_SIZE, mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].cpu().long()

@torch.no_grad()
def segment_for_eval(img_pil):
    return get_segmentation(img_pil)

def get_static_mask(seg_map):
    wt = STATIC_W.clone()
    threshold = wt.mean() + STATIC_MASK_K * wt.std()
    present = seg_map.unique().tolist()
    selected = [i for i in range(19) if wt[i] > threshold and i in present]
    if not selected:
        selected = [max([(wt[i].item(), i) for i in present if i < 19])[1]]
    mask = torch.zeros_like(seg_map, dtype=torch.float32)
    for c in selected: mask[seg_map == c] = 1.0
    return mask.numpy(), [CLASS_NAMES[c] for c in selected]

def get_gradient_mask(model, img224, label, seg_map, use_clip_norm=False):
    tf = clip_tf if use_clip_norm else resnet_tf
    x = tf(img224).unsqueeze(0).to(DEVICE).requires_grad_(True)
    model.eval(); out = model(x); out[0,label].backward()
    gm = x.grad.abs()[0].mean(0).detach().cpu()
    H,W = seg_map.shape
    gf = F.interpolate(gm.unsqueeze(0).unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False).squeeze()
    threshold = torch.quantile(gf.flatten(), 1.0-DENSITY_TOP_RATIO)
    high = (gf >= threshold)
    density = torch.zeros(19)
    for c in range(19):
        cm = (seg_map==c); a = cm.sum().float()
        if a>0: density[c] = (cm&high).sum().float()/a
    if density.max()>0: density = density/density.max()
    scores,idx = torch.topk(density, k=min(DENSITY_TOP_K,(density>0).sum().item()))
    selected = [i.item() for i in idx if density[i]>0]
    present = seg_map.unique().tolist()
    selected = [i for i in selected if i in present]
    if not selected: selected = [idx[0].item()]
    mask = torch.zeros_like(seg_map, dtype=torch.float32)
    for c in selected: mask[seg_map==c] = 1.0
    return mask.numpy(), [CLASS_NAMES[c] for c in selected]

def apply_blur(img_pil, mask_np, sigma):
    img = np.array(img_pil).astype(np.float32)
    blur = np.array(img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))).astype(np.float32)
    m = mask_np[:,:,None]
    return Image.fromarray((img*(1-m)+blur*m).astype(np.uint8))

def apply_mask_fill(img_pil, mask_np, fill_rgb):
    img = np.array(img_pil).astype(np.float32)
    m = mask_np[:,:,None]
    fill = np.array(fill_rgb, dtype=np.float32).reshape(1,1,3)
    return Image.fromarray((img*(1-m)+fill*m).astype(np.uint8))

def compute_quality(orig, out):
    if orig.size != out.size: orig = orig.resize(out.size, Image.BICUBIC)
    o = np.array(orig.convert('RGB')).astype(np.float32)/255.0
    a = np.array(out.convert('RGB')).astype(np.float32)/255.0
    return (calc_psnr(o,a,data_range=1.0), calc_ssim(o,a,data_range=1.0,channel_axis=2))

@torch.no_grad()
def classify(model, img_pil, use_clip_norm=False):
    tf = clip_tf if use_clip_norm else resnet_tf
    return model(tf(img_pil).unsqueeze(0).to(DEVICE)).argmax(1).item()

def compute_seg_metrics(so, sa, n=19):
    pa = (so==sa).float().mean().item()
    ious,per = [],{}
    present = so.unique().tolist()
    for c in range(n):
        if c not in present: continue
        pc=(sa==c); gc=(so==c)
        inter=(pc&gc).sum().float().item(); uni=(pc|gc).sum().float().item()
        iou = inter/uni if uni>0 else 1.0
        ious.append(iou); per[CITYSCAPES_CLASSES[c]]=round(iou,4)
    return pa, (float(np.mean(ious)) if ious else 1.0), per

def img_to_lpips(img_pil):
    a = np.array(img_pil.convert('RGB')).astype(np.float32)/255.0*2-1
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def run_fkl(images, exp_name):
    adv_dir = os.path.join(OUTPUT_ROOT, exp_name, 'adv')
    if not os.path.isdir(adv_dir): return None
    fid_metric.reset(); kid_metric.reset()
    lps,miss,valid = [],0,0
    for it in tqdm(images, desc=f'fkl {exp_name}'):
        op = Image.open(it['path']).convert('RGB')
        ap = os.path.join(adv_dir, it['fname'])
        if not os.path.exists(ap): miss+=1; continue
        adv = Image.open(ap).convert('RGB')
        if op.size != adv.size: op = op.resize(adv.size, Image.BICUBIC)
        fid_metric.update(fid_tf(op).unsqueeze(0).to(DEVICE), real=True)
        fid_metric.update(fid_tf(adv).unsqueeze(0).to(DEVICE), real=False)
        kid_metric.update(fid_tf(op).unsqueeze(0).to(DEVICE), real=True)
        kid_metric.update(fid_tf(adv).unsqueeze(0).to(DEVICE), real=False)
        lps.append(lpips_fn(img_to_lpips(op), img_to_lpips(adv)).item()); valid+=1
    if valid==0: return None
    fv = fid_metric.compute().item(); kv,_ = kid_metric.compute(); kv=kv.item()
    lv = float(np.mean(lps))
    print(f"  [{exp_name}] FID={fv:.4f} KID={kv:.6f} LPIPS={lv:.4f} n={valid}")
    return {'fid':round(fv,4),'kid':round(kv,6),'lpips':round(lv,4),'n':valid,'missing':miss}

def collect():
    om={}
    with open(ORIG_CSV) as f:
        for r in csv.DictReader(f): om[os.path.basename(r['path'])]=r['path']
    imgs=[]
    with open(CSV_640) as f:
        for r in csv.DictReader(f):
            fn=os.path.basename(r['path'])
            if fn in om: imgs.append({'path':r['path'],'orig_path':om[fn],'city':r['city'],'label':int(r['label']),'fname':fn})
    return imgs[START_IDX:MAX_IMAGES]

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    images = collect()
    print(f"Test images: {len(images)}")
    seg_cache={}; orig_seg={}; all_sum=[]; sem_res={}
    for exp in EXPERIMENTS:
        name,mode,region,sigma,fill = exp['name'],exp['mode'],exp['region'],exp['sigma'],exp['fill']
        adv_dir = os.path.join(OUTPUT_ROOT, name, 'adv'); os.makedirs(adv_dir, exist_ok=True)
        print(f"\n{'='*55}\n{name} mode={mode} region={region} fill={fill}\n{'='*55}")
        records,t0 = [],time.time()
        for it in tqdm(images, desc=name):
            fn=it['fname']; img=Image.open(it['path']).convert('RGB'); label=it['label']
            if fn in seg_cache: seg=seg_cache[fn]
            else:
                seg=get_segmentation(Image.open(it['orig_path']).convert('RGB'))
                seg_cache[fn]=seg; orig_seg[fn]=seg.clone()
            if region=='full':
                mask=np.ones(ATTACK_SIZE,dtype=np.float32); selected=['all']
            elif region=='static':
                mask,selected=get_static_mask(seg)
            else:
                use_clip=(exp['gmodel']=='clip'); gmodel=clip_ft if use_clip else resnet
                mask,selected=get_gradient_mask(gmodel, img.resize((224,224),Image.BICUBIC), label, seg, use_clip_norm=use_clip)
            mr=float(mask.mean())
            if mode=='blur': out=apply_blur(img,mask,sigma)
            else: out=apply_mask_fill(img,mask,FILL[fill])
            psnr,ssim=compute_quality(img,out)
            ro=classify(resnet,img); ra=classify(resnet,out)
            co=classify(clip_ft,img,True); ca=classify(clip_ft,out,True)
            records.append({'city':it['city'],'fname':fn,'psnr':float(psnr),'ssim':float(ssim),
                'mask_ratio':mr,'selected':selected,'resnet_orig':int(ro==label),'resnet_out':int(ra==label),
                'clip_orig':int(co==label),'clip_out':int(ca==label)})
            out.save(os.path.join(adv_dir,fn))
        n=len(records)
        ap=np.mean([r['psnr'] for r in records]); as_=np.mean([r['ssim'] for r in records])
        amr=np.mean([r['mask_ratio'] for r in records])
        roa=sum(r['resnet_orig'] for r in records)/n; raa=sum(r['resnet_out'] for r in records)/n
        coa=sum(r['clip_orig'] for r in records)/n; caa=sum(r['clip_out'] for r in records)/n
        rd,cd=roa-raa,coa-caa
        summ=dict(experiment=name,n=n,mode=mode,region=region,sigma=sigma,fill=fill,
            psnr=ap,ssim=as_,mask_ratio=amr,resnet_orig_acc=roa,resnet_out_acc=raa,
            resnet_drop=rd,resnet_eff=rd/amr if amr>0 else 0,clip_orig_acc=coa,clip_out_acc=caa,
            clip_drop=cd,clip_eff=cd/amr if amr>0 else 0,elapsed_min=(time.time()-t0)/60)
        all_sum.append(summ)
        json.dump({'summary':summ,'detail':records}, open(os.path.join(OUTPUT_ROOT,name,'summary.json'),'w'), indent=2)
        print(f"  PSNR={ap:.2f} MaskR={amr:.3f} CLIP:{coa:.4f}->{caa:.4f} drop={cd:.4f} eff={cd/amr if amr>0 else 0:.4f}")
    json.dump(all_sum, open(os.path.join(OUTPUT_ROOT,'all_summaries.json'),'w'), indent=2)

    # 语义一致性
    print(f"\n{'='*60}\n语义一致性\n{'='*60}")
    for exp in EXPERIMENTS:
        en=exp['name']; adv_dir=os.path.join(OUTPUT_ROOT,en,'adv')
        if not os.path.isdir(adv_dir): continue
        pas,mis,miss=[],[],0; cls={c:[] for c in CITYSCAPES_CLASSES}
        for it in tqdm(images,desc=en):
            ap=os.path.join(adv_dir,it['fname'])
            if not os.path.exists(ap): miss+=1; continue
            pa,mi,per=compute_seg_metrics(orig_seg[it['fname']], segment_for_eval(Image.open(ap).convert('RGB')))
            pas.append(pa); mis.append(mi)
            for k,v in per.items(): cls[k].append(v)
        if not pas: continue
        sem_res[en]={'pixel_accuracy':round(float(np.mean(pas)),4),'mIoU':round(float(np.mean(mis)),4),
            'per_class_iou':{k:round(float(np.mean(v)),4) for k,v in cls.items() if v},'n_evaluated':len(pas),'n_missing':miss}
        print(f"  [{en}] pixAcc={sem_res[en]['pixel_accuracy']} mIoU={sem_res[en]['mIoU']}")
    json.dump(sem_res, open(os.path.join(OUTPUT_ROOT,'semantic_consistency.json'),'w'), indent=2)

    # FID/KID/LPIPS
    load_eval_models()
    print(f"\n{'='*60}\nFID/KID/LPIPS\n{'='*60}")
    fid_res={}
    for exp in EXPERIMENTS:
        r=run_fkl(images, exp['name'])
        if r: fid_res[exp['name']]=r
    json.dump(fid_res, open(os.path.join(OUTPUT_ROOT,'results_fid_kid_lpips.json'),'w'), indent=2)
    print(f"\nSaved to {OUTPUT_ROOT}")

if __name__=='__main__':
    main()