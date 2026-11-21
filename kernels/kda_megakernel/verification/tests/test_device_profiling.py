"""GPU identity, NCU metric availability and half-instruction attribution."""
from types import SimpleNamespace

import pytest

from kernels.kda_megakernel.profiling import counters
from kernels.kda_megakernel.profiling.device import hardware_peaks


@pytest.mark.parametrize('model,sms,gib,tensor,bandwidth', [
    ('NVIDIA H100 PCIe', 114, 79, 756, 2039),
    ('NVIDIA H20', 78, 95, 148, 4000),
])
def test_full_device_peaks(tmp_path, model, sms, gib, tensor, bandwidth):
    metadata = dict(device=model, sm_count=sms, total_memory=gib << 30, compute_capability=[9, 0])
    profile = hardware_peaks(tmp_path, metadata)
    assert profile['tensor_fp16_bf16_dense_TFLOP_s'] == tensor
    assert profile['dram_total_GB_s'] == bandwidth
    assert hardware_peaks(tmp_path, dict(metadata, total_memory=10 << 30)) == {}


def test_query_selected_device_and_partial_counters(monkeypatch):
    write = 'dram__bytes_write.sum'
    read = 'dram__bytes_read.sum'
    output = '\n'.join(['Metric name,Metric type,Metric unit,Metric description,gh100',
                        f'{counters.DURATION},Counter,ns,time,1',
                        f'{write},Counter,byte,traffic,1',
                        'dram__bytes_read.avg,Counter,byte,traffic,1'])

    def query(command, **kwargs):
        assert command[command.index('--devices') + 1] == '1'
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(counters.subprocess, 'run', query)
    metrics, _ = counters.query_metrics(dict(device='NVIDIA H20', cuda_device_index=1),
                                        [counters.DURATION, write, read])
    assert metrics == [counters.DURATION, write]
    rows = counters.memory({write: 0}, 1, {'dram_total_GB_s': 4000})
    assert [row['utilization_pct'] for row in rows[:3]] == [None, 0, None]


@pytest.mark.parametrize('model', ['NVIDIA H20', 'NVIDIA H100 PCIe'])
def test_hopper_half_instruction_decoding(model):
    functions = SimpleNamespace(num_instances=lambda: 1, value=lambda index: 16)
    action = SimpleNamespace(metric_by_name=lambda name: functions,
                             sass_by_pc=lambda pc: 'HFMA2.MMA R0, R1, R2, R3' if pc == 16 else '')
    counts = {f'smsp__sass_thread_inst_executed_op_{op}_pred_on.sum': 0 for op in counters.HALF_OPS}
    counts['smsp__sass_thread_inst_executed_op_hfma_pred_on.sum'] = 3
    flops, _ = counters.half_operations(action, counts, {'device': model})
    assert flops == {'FP16': 12, 'BF16': 0}
    unknown, _ = counters.half_operations(action, counts, {'compute_capability': [10, 0]})
    assert unknown == {'FP16': None, 'BF16': None}
