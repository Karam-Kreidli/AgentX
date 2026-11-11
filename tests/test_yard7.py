"""YARD-7: yard.space_order. With least_left, scarce yard space goes first to the ship with the fewest
boxes still to place (so a ship near the end finishes); with arrival, first come first served."""
from core import params
from tests.test_coordinator import C, boxes, by_call, no_forecast, ship, world, yard_req


def _two_ships_competing(monkeypatch, order):
    no_forecast(monkeypatch)
    monkeypatch.setattr(params, "get", lambda k, _real=params.get: order if k == "yard.space_order" else _real(k))
    # Early arrives first and still has 80 boxes to place; Late has only 20 left. 30 slots free.
    st = world([ship("Early", imports=80, order_eta=0), ship("Late", berth="B2", imports=20, order_eta=1)],
               boxes("Early", 80) + boxes("Late", 20), free={"A": 30})
    return by_call(C.decide([yard_req("Early", 80), yard_req("Late", 20)], st)[0])


def test_arrival_serves_the_first_ship_first(monkeypatch):
    g = _two_ships_competing(monkeypatch, "arrival")
    assert g["Early"].granted == 30 and g["Late"].granted == 0


def test_least_left_lets_the_nearly_done_ship_finish(monkeypatch):
    g = _two_ships_competing(monkeypatch, "least_left")
    assert g["Late"].granted == 20 and g["Early"].granted == 10


def test_least_left_never_hands_out_more_than_is_free(monkeypatch):
    g = _two_ships_competing(monkeypatch, "least_left")
    assert sum(x.granted for x in g.values()) == 30


def test_the_committed_config_uses_least_left():
    from pathlib import Path
    vals = params.load(Path(__file__).resolve().parents[1] / "config" / "params.yaml")
    assert vals["yard.space_order"] == "least_left"
