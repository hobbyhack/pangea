"""
Entry point for running Pangea as a module: python -m pangea
"""

from pangea.simulation import Simulation


def main() -> None:
    """Launch the Pangea Evolution Simulator."""
    sim = Simulation()
    sim.run()


if __name__ == "__main__":
    main()
