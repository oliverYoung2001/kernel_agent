"""Compute-node device identity and offline resolution of theoretical limits."""
import json
import re


def device_identity():
    import torch

    index = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(index)
    return dict(device=prop.name, uuid=str(prop.uuid), total_memory=prop.total_memory,
                sm_count=prop.multi_processor_count, compute_capability=[prop.major, prop.minor],
                cuda_device_index=index)


def model_name(metadata):
    return re.sub(r'[\s_-]+', ' ', (metadata.get('device') or '').lower()).removeprefix('nvidia ').strip()


def hopper(metadata):
    capability = metadata.get('compute_capability')
    if capability is not None:
        return tuple(capability) == (9, 0)
    # Older captures only recorded device names.
    return model_name(metadata) in ('h100 pcie', 'h20', 'h20 96gb', 'h20 sxm5 96gb')


def hardware_peaks(experiment, metadata):
    if 'hardware_peaks' in metadata:
        profile = metadata['hardware_peaks']
        if profile and (profile['model'] != metadata['device']
                        or profile.get('sm_count', metadata.get('sm_count')) != metadata.get('sm_count')):
            raise ValueError('Recorded peak specification does not match the measured GPU')
        return profile
    gib = metadata.get('total_memory', 0) / 2**30
    matches = []
    for directory in ('experiments', 'environment'):
        for path in sorted((experiment / directory).glob('hardware*.json')):
            profile = json.loads(path.read_text())
            if not metadata.get('device') or profile.get('model') != metadata['device']:
                continue
            if 'sm_count' in profile and profile['sm_count'] != metadata.get('sm_count'):
                continue
            nominal = profile.get('memory_GB_advertised')
            if nominal and gib and not 0.92 * nominal <= gib <= nominal:
                continue
            if profile.get('uuid', metadata.get('uuid')) != metadata.get('uuid'):
                continue
            matches.append(profile)
    if matches:
        return matches[0] if len(matches) == 1 else {}

    # Match full-device SKUs only. CUDA reports usable memory, slightly below nominal.
    name = model_name(metadata)
    sms = metadata.get('sm_count')
    if hopper(metadata) and name == 'h100 pcie' and sms == 114 and 74 <= gib <= 80:
        return dict(model=metadata['device'], sm_count=sms, cuda_fp32_TFLOP_s=51.2,
                    tensor_fp16_bf16_dense_TFLOP_s=756, tensor_tf32_dense_TFLOP_s=378,
                    cuda_fp64_TFLOP_s=25.6, tensor_fp64_TFLOP_s=51.2,
                    cuda_fp16_bf16_TFLOP_s=102.4, dram_total_GB_s=2039,
                    source='https://dam-cdn.nvd.orangelogic.com/AssetLink/705n6ur546g0uk43w0117r17n8042d73.pdf')
    if hopper(metadata) and name in ('h20', 'h20 96gb', 'h20 sxm5 96gb') and sms == 78 and 89 <= gib <= 96:
        return dict(model=metadata['device'], sm_count=sms, cuda_fp32_TFLOP_s=44,
                    tensor_fp16_bf16_dense_TFLOP_s=148, tensor_tf32_dense_TFLOP_s=74,
                    cuda_fp64_TFLOP_s=1, dram_total_GB_s=4000,
                    source='https://www.an-link.com/Se_d_gci_47_id_13.html')
    return {}
