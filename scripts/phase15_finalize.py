"""Phase 15 independent held-out evaluation, run ONCE after training is finished."""
from __future__ import annotations

import argparse
import json

from holdem_ai.experiments.longrun import finalize_longrun


def build_parser():
    p=argparse.ArgumentParser(description='Seal Phase15 best checkpoint and run held-out audit')
    p.add_argument('--root',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    return p


def main():
    args=build_parser().parse_args()
    print(json.dumps(finalize_longrun(args.root,device=args.device),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
