"""Execute and validate the notebook with a project-local Jupyter kernel."""
from pathlib import Path
import os, sys, json
root = Path(__file__).resolve().parent
os.chdir(root)
packages = str(root / '.analysis_packages')
sys.path.insert(0, packages)
os.environ['PYTHONPATH'] = packages
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.jupyter_runtime')
os.environ['IPYTHONDIR'] = str(root / '.ipython')
os.environ['MPLCONFIGDIR'] = str(root / '.matplotlib_cache')
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import nbformat
kernel_root = root / '.jupyter_kernels'
kernel = kernel_root / 'analysis'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Local analysis', 'language': 'python',
    'env': {'PYTHONPATH': packages}}), encoding='utf-8')
path = root / (sys.argv[1] if len(sys.argv) > 1 else 'ORIE5260_Futures_Analysis.ipynb')
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
def report_progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Completed code cell at notebook index {cell_index}', flush=True)
client = NotebookClient(nb, timeout=600, kernel_name='analysis',
    resources={'metadata': {'path': str(root)}},
    kernel_manager_class='jupyter_client.manager.KernelManager',
    on_cell_executed=report_progress)
client.create_kernel_manager()
client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_root)])
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print('Executed and saved', path)
for cell in nb.cells:
    if cell.cell_type == 'code':
        print('Cell', cell.execution_count, 'outputs', len(cell.outputs))
