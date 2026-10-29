"""Allow ``python -m pando.execute`` to run one executor round."""
from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
