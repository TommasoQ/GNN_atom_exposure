"""
Data validation utilities for the protein atom exposure prediction dataset.
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def validate_depth_indexes(depth_path: Path) -> Tuple[bool, Dict]:
    """
    Validate the depth_indexes.pkl file.

    Returns:
        (is_valid, info_dict)
    """
    print("\n" + "="*80)
    print("VALIDATING depth_indexes.pkl")
    print("="*80)

    info = {}
    is_valid = True

    try:
        with open(depth_path, 'rb') as f:
            depth_indexes = pickle.load(f)

        # Check type
        info['type'] = str(type(depth_indexes))
        print(f"Type: {info['type']}")

        if isinstance(depth_indexes, pd.DataFrame):
            print("[OK] depth_indexes is a DataFrame")
            info['shape'] = depth_indexes.shape
            info['columns'] = list(depth_indexes.columns)
            info['num_proteins'] = depth_indexes['pdb_id'].nunique()
            info['total_atoms'] = len(depth_indexes)

            print(f"  Shape: {info['shape']}")
            print(f"  Columns: {info['columns']}")
            print(f"  Unique proteins: {info['num_proteins']}")
            print(f"  Total atoms: {info['total_atoms']:,}")

            # Validate depth values
            depths = depth_indexes['depth_index']
            info['depth_stats'] = {
                'mean': float(depths.mean()),
                'std': float(depths.std()),
                'min': float(depths.min()),
                'max': float(depths.max()),
                'has_nan': bool(depths.isna().any()),
                'has_inf': bool(np.isinf(depths).any())
            }

            print(f"\n  Depth statistics:")
            print(f"    Mean: {info['depth_stats']['mean']:.4f}")
            print(f"    Std:  {info['depth_stats']['std']:.4f}")
            print(f"    Min:  {info['depth_stats']['min']:.4f}")
            print(f"    Max:  {info['depth_stats']['max']:.4f}")
            print(f"    Has NaN: {info['depth_stats']['has_nan']}")
            print(f"    Has Inf: {info['depth_stats']['has_inf']}")

            if info['depth_stats']['has_nan'] or info['depth_stats']['has_inf']:
                print("  [ERROR] depth_index contains NaN or Inf values!")
                is_valid = False
            else:
                print("  [OK] No NaN or Inf in depth values")

        else:
            print(f"[ERROR] Expected DataFrame, got {type(depth_indexes)}")
            is_valid = False

    except Exception as e:
        print(f"[ERROR] Failed to load depth_indexes: {e}")
        info['error'] = str(e)
        is_valid = False

    return is_valid, info


def validate_protein_list(protein_csv_path: Path, depth_indexes_info: Dict) -> Tuple[bool, Dict]:
    """
    Validate the protein list CSV and match with depth_indexes.

    Returns:
        (is_valid, info_dict)
    """
    print("\n" + "="*80)
    print("VALIDATING protein_sample_5000.csv")
    print("="*80)

    info = {}
    is_valid = True

    try:
        protein_df = pd.read_csv(protein_csv_path)
        info['total_proteins'] = len(protein_df)
        print(f"Total proteins in CSV: {info['total_proteins']}")

        if 'pdb_id' not in protein_df.columns:
            print("[ERROR] 'pdb_id' column not found!")
            is_valid = False
            return is_valid, info

        # Load depth_indexes to check matching
        depth_path = protein_csv_path.parent / 'depth_indexes.pkl'
        with open(depth_path, 'rb') as f:
            depth_indexes_df = pickle.load(f)

        proteins_with_depth = set(depth_indexes_df['pdb_id'].unique())
        proteins_in_csv = set(protein_df['pdb_id'])

        proteins_with_labels = proteins_in_csv & proteins_with_depth
        proteins_missing_labels = proteins_in_csv - proteins_with_depth

        info['proteins_with_labels'] = len(proteins_with_labels)
        info['proteins_missing_labels'] = len(proteins_missing_labels)
        info['missing_percentage'] = 100 * len(proteins_missing_labels) / len(proteins_in_csv)

        print(f"\n  Proteins with depth labels: {info['proteins_with_labels']} ({100-info['missing_percentage']:.1f}%)")
        print(f"  Proteins missing labels: {info['proteins_missing_labels']} ({info['missing_percentage']:.1f}%)")

        if info['proteins_missing_labels'] > 0:
            print(f"\n  First 10 missing: {list(proteins_missing_labels)[:10]}")
            print("  [WARNING] Some proteins will be excluded from training")
        else:
            print("  [OK] All proteins have depth labels")

    except Exception as e:
        print(f"[ERROR] Failed to validate protein list: {e}")
        info['error'] = str(e)
        is_valid = False

    return is_valid, info


def validate_sample_protein(
    pdb_id: str,
    sadic_dir: Path,
    depth_indexes_df: pd.DataFrame
) -> Tuple[bool, Dict]:
    """
    Validate a single protein's data files and matching.

    Returns:
        (is_valid, info_dict)
    """
    print(f"\n" + "="*80)
    print(f"VALIDATING SAMPLE PROTEIN: {pdb_id}")
    print("="*80)

    info = {'pdb_id': pdb_id}
    is_valid = True

    try:
        protein_dir = sadic_dir / pdb_id
        if not protein_dir.exists():
            print(f"[ERROR] Protein directory not found: {protein_dir}")
            info['error'] = 'directory_not_found'
            return False, info

        # Load nodes
        nodes_path = protein_dir / f'{pdb_id}__graphein__ATOM_nodes.csv'
        if not nodes_path.exists():
            print(f"[ERROR] Nodes file not found: {nodes_path}")
            info['error'] = 'nodes_file_not_found'
            return False, info

        nodes_df = pd.read_csv(nodes_path, index_col=0)
        info['num_atoms'] = len(nodes_df)
        info['num_features'] = len(nodes_df.columns)
        print(f"  Atoms: {info['num_atoms']}")
        print(f"  Features: {info['num_features']}")

        # Check required columns
        required_cols = ['original_index', 'chain_id', 'residue_name',
                        'residue_number', 'atom_type', 'element_symbol',
                        'x_coord', 'y_coord', 'z_coord']
        missing_cols = [col for col in required_cols if col not in nodes_df.columns]

        if missing_cols:
            print(f"  [ERROR] Missing required columns: {missing_cols}")
            info['missing_columns'] = missing_cols
            is_valid = False
        else:
            print(f"  [OK] All required columns present")

        # Check for NaN in coordinates
        coord_cols = ['x_coord', 'y_coord', 'z_coord']
        nan_coords = nodes_df[coord_cols].isna().sum().sum()
        if nan_coords > 0:
            print(f"  [ERROR] {nan_coords} NaN values in coordinates")
            is_valid = False
        else:
            print(f"  [OK] No NaN in coordinates")

        # Check depth matching
        protein_depth = depth_indexes_df[depth_indexes_df['pdb_id'] == pdb_id]
        info['depth_entries'] = len(protein_depth)
        print(f"  Depth entries: {info['depth_entries']}")

        depth_dict = dict(zip(protein_depth['atom_name'], protein_depth['depth_index']))

        matched = 0
        unmatched = []
        for atom_name in nodes_df['original_index']:
            if atom_name in depth_dict:
                matched += 1
            else:
                unmatched.append(atom_name)

        info['matched_atoms'] = matched
        info['unmatched_atoms'] = len(unmatched)
        info['match_percentage'] = 100 * matched / info['num_atoms']

        print(f"  Matched atoms: {matched} ({info['match_percentage']:.1f}%)")
        print(f"  Unmatched atoms: {len(unmatched)}")

        if len(unmatched) > 0:
            print(f"  [WARNING] First 5 unmatched: {unmatched[:5]}")
            if len(unmatched) > info['num_atoms'] * 0.1:
                print(f"  [ERROR] >10% atoms unmatched!")
                is_valid = False
        else:
            print(f"  [OK] Perfect atom matching")

        # Load edges
        edges_path = protein_dir / f'{pdb_id}__graphein__ATOM_edges.csv'
        if edges_path.exists():
            edges_df = pd.read_csv(edges_path)
            info['num_edges'] = len(edges_df)
            print(f"  Edges: {info['num_edges']}")
        else:
            print(f"  [WARNING] No edges file found")

    except Exception as e:
        print(f"[ERROR] Failed to validate protein {pdb_id}: {e}")
        info['error'] = str(e)
        is_valid = False

    return is_valid, info


def validate_dataset_loading(dataset_class, root_dir: Path, num_samples: int = 5) -> Tuple[bool, Dict]:
    """
    Test loading the dataset and validate outputs.

    Args:
        dataset_class: The Dataset class to test
        root_dir: Root directory of dataset
        num_samples: Number of samples to load and validate

    Returns:
        (is_valid, info_dict)
    """
    print("\n" + "="*80)
    print(f"TESTING DATASET LOADING ({num_samples} samples)")
    print("="*80)

    info = {}
    is_valid = True

    try:
        # Initialize dataset
        print("\nInitializing dataset...")
        dataset = dataset_class(root=root_dir, normalize_features=False)

        info['dataset_size'] = len(dataset)
        print(f"  Dataset size: {info['dataset_size']}")

        if info['dataset_size'] == 0:
            print("[ERROR] Dataset is empty!")
            return False, info

        # Load and validate samples
        print(f"\nLoading {num_samples} random samples...")
        sample_info = []

        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        for i, idx in enumerate(indices):
            try:
                data = dataset[idx]

                sample = {
                    'index': int(idx),
                    'num_nodes': int(data.num_nodes),
                    'num_edges': int(data.num_edges),
                    'num_features': int(data.x.shape[1]) if data.x is not None else 0,
                    'has_targets': data.y is not None,
                    'target_shape': tuple(data.y.shape) if data.y is not None else None,
                    'has_nan_features': bool(torch.isnan(data.x).any()) if data.x is not None else False,
                    'has_inf_features': bool(torch.isinf(data.x).any()) if data.x is not None else False,
                    'has_nan_targets': bool(torch.isnan(data.y).any()) if data.y is not None else False,
                    'has_inf_targets': bool(torch.isinf(data.y).any()) if data.y is not None else False,
                }

                if data.y is not None:
                    sample['target_mean'] = float(data.y.mean())
                    sample['target_std'] = float(data.y.std())
                    sample['target_min'] = float(data.y.min())
                    sample['target_max'] = float(data.y.max())

                sample_info.append(sample)

                print(f"\n  Sample {i+1}/{num_samples} (idx={idx}):")
                print(f"    Nodes: {sample['num_nodes']}, Edges: {sample['num_edges']}")
                print(f"    Features: {sample['num_features']}")
                print(f"    Target shape: {sample['target_shape']}")

                if sample['has_nan_features'] or sample['has_inf_features']:
                    print(f"    [ERROR] Features contain NaN or Inf!")
                    is_valid = False
                else:
                    print(f"    [OK] Features clean")

                if sample['has_nan_targets'] or sample['has_inf_targets']:
                    print(f"    [ERROR] Targets contain NaN or Inf!")
                    is_valid = False
                else:
                    print(f"    [OK] Targets clean")

                if data.y is not None:
                    print(f"    Target stats: mean={sample['target_mean']:.4f}, "
                          f"std={sample['target_std']:.4f}, "
                          f"range=[{sample['target_min']:.4f}, {sample['target_max']:.4f}]")

            except Exception as e:
                print(f"\n  Sample {i+1}/{num_samples} (idx={idx}):")
                print(f"    [ERROR] Failed to load: {e}")
                sample_info.append({'index': int(idx), 'error': str(e)})
                is_valid = False

        info['samples'] = sample_info

        # Check consistency across samples
        if len(sample_info) > 0 and 'num_features' in sample_info[0]:
            feature_dims = [s['num_features'] for s in sample_info if 'num_features' in s]
            if len(set(feature_dims)) > 1:
                print(f"\n[ERROR] Inconsistent feature dimensions: {feature_dims}")
                is_valid = False
            else:
                print(f"\n[OK] Consistent feature dimensions: {feature_dims[0]}")
                info['feature_dim'] = feature_dims[0]

    except Exception as e:
        print(f"[ERROR] Failed to test dataset loading: {e}")
        import traceback
        traceback.print_exc()
        info['error'] = str(e)
        is_valid = False

    return is_valid, info


def run_full_validation(root_dir: Path, dataset_class=None) -> Dict:
    """
    Run full validation pipeline.

    Args:
        root_dir: Root directory containing dataset/
        dataset_class: Optional dataset class to test loading

    Returns:
        validation_report: Dict with all validation results
    """
    print("\n" + "#"*80)
    print("STARTING FULL DATA VALIDATION")
    print("#"*80)

    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'root_dir': str(root_dir),
        'validations': {}
    }

    # Validate depth_indexes
    depth_path = root_dir / 'dataset' / 'depth_indexes.pkl'
    is_valid, info = validate_depth_indexes(depth_path)
    report['validations']['depth_indexes'] = {
        'is_valid': is_valid,
        'info': info
    }

    # Validate protein list
    protein_csv = root_dir / 'dataset' / 'protein_sample_5000.csv'
    is_valid, info = validate_protein_list(protein_csv, report['validations']['depth_indexes']['info'])
    report['validations']['protein_list'] = {
        'is_valid': is_valid,
        'info': info
    }

    # Validate sample proteins
    depth_path = root_dir / 'dataset' / 'depth_indexes.pkl'
    with open(depth_path, 'rb') as f:
        depth_indexes_df = pickle.load(f)

    # Test 3 sample proteins
    sample_pdb_ids = list(depth_indexes_df['pdb_id'].unique())[:3]
    sadic_dir = root_dir / 'dataset' / 'sadic_data'

    report['validations']['sample_proteins'] = {}
    for pdb_id in sample_pdb_ids:
        is_valid, info = validate_sample_protein(pdb_id, sadic_dir, depth_indexes_df)
        report['validations']['sample_proteins'][pdb_id] = {
            'is_valid': is_valid,
            'info': info
        }

    # Test dataset loading if class provided
    if dataset_class is not None:
        is_valid, info = validate_dataset_loading(dataset_class, root_dir, num_samples=5)
        report['validations']['dataset_loading'] = {
            'is_valid': is_valid,
            'info': info
        }

    # Overall validation status
    report['overall_valid'] = all(
        v['is_valid']
        for v in report['validations'].values()
        if isinstance(v, dict) and 'is_valid' in v
    ) and all(
        v['is_valid']
        for v in report['validations'].get('sample_proteins', {}).values()
    )

    print("\n" + "#"*80)
    print("VALIDATION SUMMARY")
    print("#"*80)
    print(f"\nOverall Status: {'PASS' if report['overall_valid'] else 'FAIL'}")
    print(f"\nIndividual Checks:")
    print(f"  depth_indexes: {'PASS' if report['validations']['depth_indexes']['is_valid'] else 'FAIL'}")
    print(f"  protein_list: {'PASS' if report['validations']['protein_list']['is_valid'] else 'FAIL'}")
    print(f"  sample_proteins: {sum(v['is_valid'] for v in report['validations']['sample_proteins'].values())}/{len(report['validations']['sample_proteins'])} passed")
    if 'dataset_loading' in report['validations']:
        print(f"  dataset_loading: {'PASS' if report['validations']['dataset_loading']['is_valid'] else 'FAIL'}")

    print("\n" + "#"*80)

    return report


if __name__ == '__main__':
    # Run validation from command line
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent.parent  # Default to project root

    print(f"Validating dataset at: {root_dir}")

    # Try to import dataset class
    try:
        sys.path.insert(0, str(root_dir / 'src'))
        from data.dataset_fixed import ProteinAtomDataset
        report = run_full_validation(root_dir, dataset_class=ProteinAtomDataset)
    except ImportError:
        print("[WARNING] Could not import ProteinAtomDataset, skipping dataset loading test")
        report = run_full_validation(root_dir, dataset_class=None)

    # Save report
    report_path = root_dir / 'experiments' / 'progress' / 'validation_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nValidation report saved to: {report_path}")

    sys.exit(0 if report['overall_valid'] else 1)
