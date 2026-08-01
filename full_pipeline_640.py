#!/usr/bin/env python3

import os, sys, csv, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
import clip
from tqdm import tqdm
from pathlib import Path
 
# ── 项目路径：换电脑时保持下面的目录结构即可，不再写死本机绝对路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent
EXTERNAL_METHODS_DIR = PROJECT_ROOT / 'external_methods'
BISENET_ROOT = EXTERNAL_METHODS_DIR / 'bisenetv2'
ATTACK_BACKEND_DIR = EXTERNAL_METHODS_DIR / 'attack_backend'
DATA_ROOT = PROJECT_ROOT / 'data' / 'cityscapes'

sys.path.insert(0, str(EXTERNAL_METHODS_DIR))
sys.path.insert(0, str(BISENET_ROOT))
sys.path.insert(0, str(ATTACK_BACKEND_DIR))

BISENET_WEIGHT = BISENET_ROOT / 'weights' / 'model_final_v2_city.pth'
BISENET_CONFIG = BISENET_ROOT / 'configs' / 'bisenetv2_city.py'
CLASSIFIER_DIR = DATA_ROOT / 'city_classifier'
OUTPUT_ROOT = DATA_ROOT / '640' / 'results_batch'

ORIG_CSV = DATA_ROOT / 'test_list.csv'
CSV_640 = DATA_ROOT / '640' / 'test_list_640.csv'
 
ATTACK_SIZE = (640, 640)
DEVICE      = torch.device('cuda')
NUM_CLASSES = 14
START_IDX  = 0
MAX_IMAGES = 1000
 
DENSITY_TOP_RATIO = 0.05
STATIC_MASK_K = -0.3
DENSITY_TOP_K = 3
PIXEL_ADAPTIVE_K  = 1.0
 
EXPERIMENTS = [
    dict(name='baseline_eps8',     mask='full',     eps=8, steps=100, gmodel=None),
    dict(name='static_eps8',      mask='static',   eps=8, steps=100, gmodel=None),
    dict(name='resnet_grad_eps8', mask='gradient', eps=8, steps=100, gmodel='resnet'),
    dict(name='clip_grad_eps8',   mask='gradient', eps=8, steps=100, gmodel='clip'),

]
 
RUN_ATTACK        = True
RUN_SEM_CONSIST   = True
RUN_FID_KID_LPIPS = True
EVAL_EXP_NAMES = [exp['name'] for exp in EXPERIMENTS]
 
from sensitivity.table import WEIGHTS as _W, CLASS_NAMES
STATIC_W = torch.tensor(_W, dtype=torch.float32)
 
CITYSCAPES_CLASSES = [
    'road','sidewalk','building','wall','fence','pole',
    'traffic light','traffic sign','vegetation','terrain',
    'sky','person','rider','car','truck','bus',
    'train','motorcycle','bicycle'
]
 
seg_tf = transforms.Compose([
    transforms.Resize((512, 1024)), transforms.ToTensor(),
    transforms.Normalize([0.3257,0.3690,0.3223],[0.2112,0.2148,0.2115])
])
resnet_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
clip_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.48145466,0.4578275,0.40821073],
                         [0.26862954,0.26130258,0.27577711])
])
 
def gs_transform(img_pil):
    img_pil = img_pil.convert('RGB')
    if img_pil.size != (ATTACK_SIZE[1], ATTACK_SIZE[0]):
        img_pil = img_pil.resize((ATTACK_SIZE[1], ATTACK_SIZE[0]), Image.BICUBIC)
    arr = np.array(img_pil).astype(np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)
 
fid_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).byte())
])
 
class CLIPFinetune(nn.Module):
    def __init__(self, cm, nc):
        super().__init__()
        self.encoder = cm.visual
        self.fc = nn.Linear(512, nc)
    def forward(self, x):
        return self.fc(self.encoder(x))
 
print("Loading BiSeNetV2 (Cityscapes)...")
from configs import set_cfg_from_file
from lib.models import model_factory

bisenet_cfg = set_cfg_from_file(str(BISENET_CONFIG))
if int(bisenet_cfg.n_cats) != 19:
    raise ValueError(
        f"BiSeNetV2配置声明{bisenet_cfg.n_cats}类，但Cityscapes实验要求19类"
    )

bisenet_model = model_factory[bisenet_cfg.model_type](
    bisenet_cfg.n_cats, aux_mode='eval'
)
bisenet_checkpoint = torch.load(BISENET_WEIGHT, map_location='cpu')
if isinstance(bisenet_checkpoint, dict):
    for checkpoint_key in ('state_dict', 'model_state_dict', 'model'):
        if checkpoint_key in bisenet_checkpoint and isinstance(
                bisenet_checkpoint[checkpoint_key], dict):
            bisenet_checkpoint = bisenet_checkpoint[checkpoint_key]
            break
if not isinstance(bisenet_checkpoint, dict):
    raise TypeError(f"无法从{BISENET_WEIGHT}读取BiSeNetV2 state_dict")
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
resnet.load_state_dict(torch.load(
    os.path.join(CLASSIFIER_DIR, 'resnet50_imagenet_best.pth'), map_location='cpu'))
resnet.eval().to(DEVICE)
 
base_clip, _ = clip.load('ViT-B/16', device='cpu')
base_clip = base_clip.float()
clip_ft = CLIPFinetune(base_clip, NUM_CLASSES)
clip_ft.load_state_dict(torch.load(
    os.path.join(CLASSIFIER_DIR, 'clip_finetune_best.pth'), map_location='cpu'))
clip_ft.eval().to(DEVICE)
 
print("Initializing attack backend...")
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir
GlobalHydra.instance().clear()
config_dir = ATTACK_BACKEND_DIR / 'config'
initialize_config_dir(version_base=None, config_dir=str(config_dir))
compose(config_name='ensemble_3models')
 
from omegaconf import OmegaConf
from geoshield import (get_models, get_ensemble_loss,
                       attack_imgpair, set_environment, to_tensor as gs_to_tensor)
set_environment()
 
cfg = OmegaConf.create({
    "data": {"batch_size":1,"num_samples":99999,"cle_data_path":'/tmp',
             "tgt_data_path":'/tmp',"output":'/tmp',"bbox_json_path":""},
    "optim": {"epsilon":8.0,"alpha":1.0,"steps":100},
    "model": {"input_res":640,"ensemble":True,"backbone":["B16","B32","Laion"],
              "device":str(DEVICE),"use_source_crop":False,"use_target_crop":False,
              "crop_scale":[0.5,0.9]},
    "wandb":{"project":"batch","entity":""},
    "attack":"fgsm",
})
 
ensemble_extractor, gs_models = get_models(cfg)
ensemble_loss = get_ensemble_loss(cfg, gs_models)
 
lpips_fn = fid_metric = kid_metric = None
def load_eval_models():
    global lpips_fn, fid_metric, kid_metric
    if lpips_fn is not None: return
    import lpips as lpips_lib
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    print("Loading LPIPS / FID / KID models...")
    lpips_fn   = lpips_lib.LPIPS(net='alex').to(DEVICE)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(DEVICE)
    kid_metric = KernelInceptionDistance(feature=2048, subset_size=50, normalize=False).to(DEVICE)
    print("Eval models loaded.")
 
def resolve_csv_image_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = DATA_ROOT / path
    return str(path)

def collect_test_images():
    orig_map = {}
    with open(ORIG_CSV) as f:
        for row in csv.DictReader(f):
            orig_path = resolve_csv_image_path(row['path'])
            orig_map[os.path.basename(orig_path)] = orig_path
    images = []
    with open(CSV_640) as f:
        for row in csv.DictReader(f):
            image_path = resolve_csv_image_path(row['path'])
            fname = os.path.basename(image_path)
            if fname not in orig_map:
                print(f"  [警告] 找不到原图: {fname}")
                continue
            images.append({'path':image_path,'orig_path':orig_map[fname],
                           'city':row['city'],'label':int(row['label']),'fname':fname})
    subset = images[START_IDX:MAX_IMAGES]
    print(f"Test images: {len(subset)} (idx {START_IDX}~{MAX_IMAGES-1})")
    return subset
 
@torch.no_grad()
def get_segmentation(img_pil):
    # BiSeNetV2直接输出Cityscapes 19类，再插值到640x640
    inp = seg_tf(img_pil.convert('RGB')).unsqueeze(0).to(DEVICE)
    out = bisenet_model(inp)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits = F.interpolate(
        logits, size=ATTACK_SIZE, mode='bilinear', align_corners=False
    )
    return logits.argmax(1)[0].cpu().long()
 
@torch.no_grad()
def segment_for_eval(img_pil):
    inp = seg_tf(img_pil.convert('RGB')).unsqueeze(0).to(DEVICE)
    out = bisenet_model(inp)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits = F.interpolate(
        logits, size=ATTACK_SIZE, mode='bilinear', align_corners=False
    )
    return logits.argmax(1)[0].cpu().long()
 
def get_static_mask(seg_map):
    wt = STATIC_W.clone()
    threshold = wt.mean() + STATIC_MASK_K * wt.std()
    present = seg_map.unique().tolist()
    selected = [i for i in range(19) if wt[i] > threshold and i in present]
    if not selected:
        selected = [max([(wt[i].item(), i) for i in present])[1]]
    mask = torch.zeros_like(seg_map, dtype=torch.float32)
    for c in selected:
        mask[seg_map == c] = 1.0
    return mask.unsqueeze(0), [CLASS_NAMES[c] for c in selected]
 
def get_density_weights(model, img_pil_224, label, seg_map, use_clip_norm=False, top_ratio=0.05):
    tf = clip_tf if use_clip_norm else resnet_tf
    x = tf(img_pil_224).unsqueeze(0).to(DEVICE).requires_grad_(True)
    model.eval()
    out = model(x)
    out[0, label].backward()
    gm = x.grad.abs()[0].mean(0).detach().cpu()
    H, W = seg_map.shape
    gf = F.interpolate(gm.unsqueeze(0).unsqueeze(0), size=(H, W),
                       mode='bilinear', align_corners=False).squeeze()
    threshold = torch.quantile(gf.flatten(), 1.0 - top_ratio)
    high_mask = (gf >= threshold)
    density = torch.zeros(19)
    for cls_id in range(19):
        cls_mask = (seg_map == cls_id)
        area = cls_mask.sum().float()
        if area > 0:
            density[cls_id] = (cls_mask & high_mask).sum().float() / area
    if density.max() > 0:
        density = density / density.max()
    return density
 
def density_to_mask(density, seg_map, top_k=4):
    scores, indices = torch.topk(density, k=min(top_k, (density > 0).sum().item()))
    selected = [i.item() for i in indices if density[i] > 0]
    present = seg_map.unique().tolist()
    selected = [i for i in selected if i in present]
    if not selected:
        selected = [indices[0].item()]
    mask = torch.zeros_like(seg_map, dtype=torch.float32)
    for c in selected:
        mask[seg_map == c] = 1.0
    return mask.unsqueeze(0), [CLASS_NAMES[c] for c in selected]
 
def get_pixel_mask(model, img_pil_224, label, seg_map, use_clip_norm=False, adaptive_k=1.0):
    tf = clip_tf if use_clip_norm else resnet_tf
    x = tf(img_pil_224).unsqueeze(0).to(DEVICE).requires_grad_(True)
    model.eval()
    out = model(x)
    out[0, label].backward()
    gm = x.grad.abs()[0].mean(0).detach().cpu()
    H, W = seg_map.shape
    gf = F.interpolate(gm.unsqueeze(0).unsqueeze(0), size=(H, W),
                       mode='bilinear', align_corners=False).squeeze()
    threshold = gf.mean() + adaptive_k * gf.std()
    mask = (gf >= threshold).float().unsqueeze(0)
    return mask, round(mask.mean().item(), 4)
 
def compute_quality(orig, adv):
    if orig.size != adv.size:
        orig = orig.resize(adv.size, Image.BICUBIC)
    o = np.array(orig.convert('RGB')).astype(np.float32) / 255.0
    a = np.array(adv.convert('RGB')).astype(np.float32) / 255.0
    return (calc_psnr(o, a, data_range=1.0),
            calc_ssim(o, a, data_range=1.0, channel_axis=2))
 
@torch.no_grad()
def classify(model, img_pil, use_clip_norm=False):
    tf = clip_tf if use_clip_norm else resnet_tf
    x = tf(img_pil).unsqueeze(0).to(DEVICE)
    return model(x).argmax(1).item()
 
def compute_seg_metrics(seg_orig, seg_adv, n_classes=19):
    pixel_acc = (seg_orig == seg_adv).float().mean().item()
    iou_list, per_class = [], {}
    present_classes = seg_orig.unique().tolist()
    for c in range(n_classes):
        if c not in present_classes: continue
        pred_c = (seg_adv == c); gt_c = (seg_orig == c)
        intersection = (pred_c & gt_c).sum().float().item()
        union = (pred_c | gt_c).sum().float().item()
        iou = intersection / union if union > 0 else 1.0
        iou_list.append(iou)
        per_class[CITYSCAPES_CLASSES[c]] = round(iou, 4)
    miou = float(np.mean(iou_list)) if iou_list else 1.0
    return pixel_acc, miou, per_class
 
def img_to_lpips_tensor(img_pil):
    arr = np.array(img_pil.convert('RGB')).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
 
@torch.no_grad()
def run_fid_kid_lpips(images, exp_name):
    adv_dir = os.path.join(OUTPUT_ROOT, exp_name, 'adv')
    if not os.path.isdir(adv_dir):
        print(f"  [SKIP FID/KID/LPIPS] {exp_name}: adv dir not found")
        return None
    fid_metric.reset(); kid_metric.reset()
    lpips_scores, missing, valid = [], 0, 0
    for item in tqdm(images, desc=f'fid/kid/lpips {exp_name}'):
        orig_pil = Image.open(item['path']).convert('RGB')
        adv_path = os.path.join(adv_dir, item['fname'])
        if not os.path.exists(adv_path):
            missing += 1; continue
        adv_pil = Image.open(adv_path).convert('RGB')
        if orig_pil.size != adv_pil.size:
            orig_pil = orig_pil.resize(adv_pil.size, Image.BICUBIC)
        orig_t = fid_tf(orig_pil).unsqueeze(0).to(DEVICE)
        adv_t  = fid_tf(adv_pil).unsqueeze(0).to(DEVICE)
        fid_metric.update(orig_t, real=True); fid_metric.update(adv_t, real=False)
        kid_metric.update(orig_t, real=True); kid_metric.update(adv_t, real=False)
        lpips_scores.append(lpips_fn(img_to_lpips_tensor(orig_pil),
                                     img_to_lpips_tensor(adv_pil)).item())
        valid += 1
    if valid == 0: return None
    fid_val = fid_metric.compute().item()
    kid_val,_ = kid_metric.compute(); kid_val = kid_val.item()
    lpips_val = float(np.mean(lpips_scores))
    print(f"  [{exp_name}] FID={fid_val:.4f}  KID={kid_val:.6f}  LPIPS={lpips_val:.4f}  n={valid}  missing={missing}")
    return {'fid':round(fid_val,4),'kid':round(kid_val,6),'lpips':round(lpips_val,4),'n':valid,'missing':missing}
 
def run_semantic_consistency(images, exp_names, orig_seg640_map):
    print(f"\n{'='*60}\n语义一致性评估（640 vs 640）\n{'='*60}")
    all_results = {}
    for exp_name in exp_names:
        adv_dir = os.path.join(OUTPUT_ROOT, exp_name, 'adv')
        if not os.path.isdir(adv_dir):
            print(f"[SKIP] {exp_name}: adv dir not found"); continue
        pixel_accs, mious, missing = [], [], 0
        class_ious = {c: [] for c in CITYSCAPES_CLASSES}
        for item in tqdm(images, desc=exp_name):
            adv_path = os.path.join(adv_dir, item['fname'])
            if not os.path.exists(adv_path):
                missing += 1; continue
            adv_pil = Image.open(adv_path).convert('RGB')
            seg_orig = orig_seg640_map[item['fname']]
            seg_adv = segment_for_eval(adv_pil)
            pa, miou, per_cls = compute_seg_metrics(seg_orig, seg_adv)
            pixel_accs.append(pa); mious.append(miou)
            for cls_name, iou in per_cls.items():
                class_ious[cls_name].append(iou)
        if not pixel_accs: continue
        avg_pa, avg_miou = float(np.mean(pixel_accs)), float(np.mean(mious))
        avg_per_class = {c: round(float(np.mean(v)), 4) for c, v in class_ious.items() if v}
        all_results[exp_name] = {'pixel_accuracy':round(avg_pa,4),'mIoU':round(avg_miou,4),
                                 'per_class_iou':avg_per_class,'n_evaluated':len(pixel_accs),'n_missing':missing}
        print(f"  [{exp_name}] pixel_acc={avg_pa:.4f}  mIoU={avg_miou:.4f}  n={len(pixel_accs)}  missing={missing}")
    out_path = os.path.join(OUTPUT_ROOT, 'semantic_consistency.json')
    with open(out_path, 'w') as f: json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")
    return all_results
 
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    images = collect_test_images()
    seg_cache = {}
    orig_seg640_map = {}
    attack_summaries = {}
    if RUN_ATTACK:
        print(f"\n{'='*60}\n阶段1：攻击实验（640）\n{'='*60}")
        for exp in EXPERIMENTS:
            name, mtype, eps, steps, gmodel = exp['name'],exp['mask'],exp['eps'],exp['steps'],exp['gmodel']
            out_dir = os.path.join(OUTPUT_ROOT, name)
            adv_dir = os.path.join(out_dir, 'adv')
            os.makedirs(adv_dir, exist_ok=True)
            print(f"\n{'='*55}\nExperiment: {name}  eps={eps} steps={steps}\n{'='*55}")
            records, t0 = [], time.time()
            for item in tqdm(images, desc=name):
                fname = item['fname']
                if fname in seg_cache:
                    seg_map = seg_cache[fname]
                else:
                    orig_pil = Image.open(item['orig_path']).convert('RGB')
                    seg_map = get_segmentation(orig_pil)   # 已是640x640
                    seg_cache[fname] = seg_map
                    orig_seg640_map[fname] = seg_map.clone()   # 直接用，已是640
                img_pil = Image.open(item['path']).convert('RGB')
                label = item['label']
                img_pil_224 = img_pil.resize((224, 224), Image.BICUBIC)
                if mtype == 'full':
                    H0, W0 = seg_map.shape
                    mask = torch.ones(1, H0, W0); selected = ['all']
                elif mtype == 'static':
                    mask, selected = get_static_mask(seg_map)
                elif mtype == 'pixel':
                    use_clip = (gmodel == 'clip')
                    grad_model = clip_ft if use_clip else resnet
                    mask, _ = get_pixel_mask(grad_model, img_pil_224, label, seg_map,
                                             use_clip_norm=use_clip, adaptive_k=PIXEL_ADAPTIVE_K)
                    selected = [f'pixel_k{PIXEL_ADAPTIVE_K}']
                else:
                    use_clip = (gmodel == 'clip')
                    grad_model = clip_ft if use_clip else resnet
                    density = get_density_weights(grad_model, img_pil_224, label, seg_map,
                                                  use_clip_norm=use_clip, top_ratio=DENSITY_TOP_RATIO)
                    mask, selected = density_to_mask(density, seg_map, top_k=DENSITY_TOP_K)
                mask_ratio = mask.mean().item()
                img_gs = gs_transform(img_pil).unsqueeze(0).to(DEVICE)
                aH, aW = ATTACK_SIZE
                mask_gs = F.interpolate(mask.float().unsqueeze(0), size=(aH, aW),
                                        mode='nearest').expand(1, 3, aH, aW).to(DEVICE)
                cfg.model.input_res = max(aH, aW)
                cfg.optim.epsilon = float(eps); cfg.optim.steps = steps
                cfg.data.output = out_dir
                cfg.data.cle_data_path = os.path.dirname(item['path'])
                cfg.data.tgt_data_path = cfg.data.cle_data_path
                attack_imgpair(cfg=cfg, ensemble_extractor=ensemble_extractor,
                    ensemble_loss=ensemble_loss, source_crop=lambda x: x,
                    target_crop=lambda x: x, img_index=0, image_org=img_gs,
                    path_org=[item['path']], image_tgt=img_gs, bbox_dict={},
                    semantic_mask=mask_gs)
                adv_path = None
                for root, dirs, files in os.walk(out_dir):
                    for f in files:
                        if f == fname:
                            adv_path = os.path.join(root, f); break
                if adv_path is None:
                    for root, dirs, files in os.walk(out_dir):
                        for f in files:
                            if (fname.split('.')[0] in f and f.endswith('.png')):
                                adv_path = os.path.join(root, f); break
                if adv_path is None:
                    print(f"  [跳过] 未找到: {fname}"); continue
                adv_pil = Image.open(adv_path).convert('RGB')
                if adv_pil.size != (aW, aH):
                    adv_pil = adv_pil.resize((aW, aH), Image.BICUBIC)
                psnr_v, ssim_v = compute_quality(img_pil, adv_pil)
                r_orig = classify(resnet, img_pil, use_clip_norm=False)
                r_adv  = classify(resnet, adv_pil, use_clip_norm=False)
                c_orig = classify(clip_ft, img_pil, use_clip_norm=True)
                c_adv  = classify(clip_ft, adv_pil, use_clip_norm=True)
                records.append({'city':item['city'],'fname':fname,'psnr':float(psnr_v),
                    'ssim':float(ssim_v),'mask_ratio':float(mask_ratio),'selected':selected,
                    'resnet_orig':int(r_orig==label),'resnet_adv':int(r_adv==label),
                    'clip_orig':int(c_orig==label),'clip_adv':int(c_adv==label)})
                adv_pil.save(os.path.join(adv_dir, fname))
            n = len(records)
            avg_psnr = np.mean([r['psnr'] for r in records])
            avg_ssim = np.mean([r['ssim'] for r in records])
            avg_mr = np.mean([r['mask_ratio'] for r in records])
            r_orig_acc = sum(r['resnet_orig'] for r in records)/n
            r_adv_acc = sum(r['resnet_adv'] for r in records)/n
            c_orig_acc = sum(r['clip_orig'] for r in records)/n
            c_adv_acc = sum(r['clip_adv'] for r in records)/n
            r_drop, c_drop = r_orig_acc-r_adv_acc, c_orig_acc-c_adv_acc
            r_eff = r_drop/avg_mr if avg_mr>0 else 0
            c_eff = c_drop/avg_mr if avg_mr>0 else 0
            summary = dict(experiment=name,n=n,eps=eps,psnr=avg_psnr,ssim=avg_ssim,
                mask_ratio=avg_mr,resnet_orig_acc=r_orig_acc,resnet_adv_acc=r_adv_acc,
                resnet_drop=r_drop,resnet_eff=r_eff,clip_orig_acc=c_orig_acc,
                clip_adv_acc=c_adv_acc,clip_drop=c_drop,clip_eff=c_eff,
                elapsed_min=(time.time()-t0)/60)
            attack_summaries[name] = summary
            with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
                json.dump({'summary':summary,'detail':records}, f, indent=2)
            print(f"\n[{name}] PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} MaskR={avg_mr:.3f}")
            print(f"  ResNet: {r_orig_acc:.4f}->{r_adv_acc:.4f} drop={r_drop:.4f} eff={r_eff:.4f}")
            print(f"  CLIP:   {c_orig_acc:.4f}->{c_adv_acc:.4f} drop={c_drop:.4f} eff={c_eff:.4f}")
        with open(os.path.join(OUTPUT_ROOT, 'all_summaries.json'), 'w') as f:
            json.dump(list(attack_summaries.values()), f, indent=2)
        W = 110
        print(f"\n{'='*W}")
        print(f"{'Experiment':<22} {'PSNR':>6} {'SSIM':>6} {'MaskR':>7} {'R-orig':>7} {'R-adv':>7} {'R-drop':>7} {'R-eff':>8} {'C-drop':>7} {'C-eff':>8}")
        print(f"{'='*W}")
        for s in attack_summaries.values():
            print(f"{s['experiment']:<22} {s['psnr']:>6.2f} {s['ssim']:>6.4f} {s['mask_ratio']:>7.3f} {s['resnet_orig_acc']:>7.4f} {s['resnet_adv_acc']:>7.4f} {s['resnet_drop']:>7.4f} {s['resnet_eff']:>8.4f} {s['clip_drop']:>7.4f} {s['clip_eff']:>8.4f}")
        print(f"{'='*W}")
    sem_results = {}
    if RUN_SEM_CONSIST:
        if not orig_seg640_map:
            print("补充干净图640分割...")
            for item in tqdm(images, desc='orig seg640'):
                orig_pil = Image.open(item['orig_path']).convert('RGB')
                orig_seg640_map[item['fname']] = get_segmentation(orig_pil)   # 已是640
        sem_results = run_semantic_consistency(images, EVAL_EXP_NAMES, orig_seg640_map)
    fid_results = {}
    if RUN_FID_KID_LPIPS:
        load_eval_models()
        print(f"\n{'='*60}\n阶段3：FID / KID / LPIPS 评估（640 vs 640）\n{'='*60}")
        for exp_name in EVAL_EXP_NAMES:
            r = run_fid_kid_lpips(images, exp_name)
            if r: fid_results[exp_name] = r
        out_path = os.path.join(OUTPUT_ROOT, 'results_fid_kid_lpips.json')
        with open(out_path, 'w') as f: json.dump(fid_results, f, indent=2)
        print(f"FID/KID/LPIPS saved to {out_path}")
    print(f"\n{'='*100}")
    print(f"{'Experiment':<22} {'PSNR':>6} {'SSIM':>6} {'MaskR':>7} {'C-drop':>7} {'C-eff':>7} {'PixAcc':>7} {'mIoU':>7} {'FID':>8} {'KID':>10} {'LPIPS':>7}")
    print(f"{'='*100}")
    for exp_name in EVAL_EXP_NAMES:
        atk = attack_summaries.get(exp_name, {})
        sem = sem_results.get(exp_name, {})
        fid = fid_results.get(exp_name, {})
        print(f"{exp_name:<22} {atk.get('psnr',0):>6.2f} {atk.get('ssim',0):>6.4f} {atk.get('mask_ratio',0):>7.3f} {atk.get('clip_drop',0):>7.4f} {atk.get('clip_eff',0):>7.4f} {sem.get('pixel_accuracy',0):>7.4f} {sem.get('mIoU',0):>7.4f} {fid.get('fid',0):>8.4f} {fid.get('kid',0):>10.6f} {fid.get('lpips',0):>7.4f}")
    print(f"{'='*100}")
    print(f"\n所有结果已保存至 {OUTPUT_ROOT}")
 
if __name__ == '__main__':
    main()