from core.evaluation import StudyRunner, load_ml100k_ratings
import importlib
import inspect
import sys
import os


def main():
    data_dir = os.path.join('movielens', 'ml-100k')
    train_file = 'ua.base'
    test_file = 'ua.test'
    print(f'Loading train: {os.path.join(data_dir, train_file)}')
    try:
        train = load_ml100k_ratings(data_dir, train_file)
        test = load_ml100k_ratings(data_dir, test_file)
    except Exception as e:
        print('Error loading MovieLens files:', e)
        print('Make sure you have the ML-100k files in movielens/ml-100k/')
        return

    base_dir = os.path.dirname(__file__)
    for pkg in ('memoryBased', 'modelBased'):
        pkg_path = os.path.join(base_dir, pkg)
        if not os.path.isdir(pkg_path):
            continue
        for fname in os.listdir(pkg_path):
            if not fname.endswith('.py') or fname.startswith('__'):
                continue
            modname = fname[:-3]
            full_mod = f"{pkg}.{modname}"
            try:
                importlib.import_module(full_mod)
            except Exception:
                # ignore modules that fail to import; registry will only contain
                # successfully imported decorated classes
                continue

    try:
        from core.recommender_registry import get_registry
        available = {k.lower(): v for k, v in get_registry().items()}
    except Exception:
        available = {}
    if len(sys.argv) > 1:
        requested = [a.lower() for a in sys.argv[1:]]
        chosen = {}
        missing = []
        for r in requested:
            name = r[:-3] if r.endswith('.py') else r
            if name in available:
                chosen[name] = available[name]
            else:
                missing.append(r)
        if missing:
            print('Unknown algorithm(s):', ', '.join(missing))
            print('Available (registered):', ', '.join(sorted(available.keys())))
            return
    else:
        chosen = available

    algos = {}
    for name, cls in chosen.items():
        try:
            inst = cls()
            if hasattr(inst, 'fit'):
                inst.fit(train)
            algos[name] = inst
        except Exception as e:
            print(f'Failed to initialize {name}: {e}')

    runner = StudyRunner()
    save_dir = os.path.join('results')
    results = runner.run(algos, train, test, k=10, save_dir=save_dir)

    print(f'\nEvaluation results (saved to {save_dir}):')
    for name, metrics in results.items():
        print(f'\n{name}:')
        for m, v in metrics.items():
            print(f'  {m}: {v:.4f}')


if __name__ == '__main__':
    main()
