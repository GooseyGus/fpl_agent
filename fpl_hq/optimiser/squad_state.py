"""Track current squad state between gameweeks."""
from pathlib import Path
import json


class SquadState:
    """Stores and loads squad composition: 15 player IDs (11 starters + 4 bench)."""
    
    def __init__(self, player_ids: list[int] | None = None, gameweek: int = 0, 
                 bank: float = 0.0, free_transfers: int = 1,
                 selling_prices: dict[int, float] | None = None):
        """
        Args:
            player_ids: List of 15 player IDs (last 4 are bench)
            gameweek: Gameweek this squad is for
            bank: Money in the bank (millions)
            free_transfers: Available free transfers
            selling_prices: Selling price (millions) per player ID; current price when absent
        """
        if player_ids:
            if len(player_ids) != 15:
                raise ValueError(f"Expected 15 players, got {len(player_ids)}")
        
        self.player_ids = player_ids or []
        self.gameweek = gameweek
        self.bank = bank
        self.free_transfers = free_transfers
        self.selling_prices = {int(pid): float(price) for pid, price in (selling_prices or {}).items()}
    
    @property
    def starters(self) -> list[int]:
        """First 11 are starters."""
        return self.player_ids[:11]
    
    @property
    def bench(self) -> list[int]:
        """Last 4 are bench."""
        return self.player_ids[11:15]
    
    def save(self, filepath: Path) -> None:
        """Save squad to JSON file."""
        data = {
            "gameweek": int(self.gameweek),
            "player_ids": [int(pid) for pid in self.player_ids],  # Convert numpy int64 to int
            "bank": float(self.bank),
            "free_transfers": int(self.free_transfers),
            "selling_prices": {str(pid): price for pid, price in self.selling_prices.items()},
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved squad for GW{self.gameweek} to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> "SquadState":
        """Load squad from JSON file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Squad file not found: {filepath}")
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return cls(
            player_ids=data["player_ids"],
            gameweek=data["gameweek"],
            bank=data["bank"],
            free_transfers=data["free_transfers"],
            selling_prices=data.get("selling_prices"),
        )
    
    def __repr__(self) -> str:
        return (f"SquadState(GW{self.gameweek}, players={len(self.player_ids)}, "
                f"bank={self.bank}M, FT={self.free_transfers})")


if __name__ == "__main__":
    # Test
    squad = SquadState(
        player_ids=list(range(1, 16)),
        gameweek=0,
        bank=5.0,
        free_transfers=1
    )
    print(squad)
    print(f"Starters: {squad.starters}")
    print(f"Bench: {squad.bench}")
