from .query import query_trade_data
from .query_rent import query_rent_data
from .nearby import query_trade_nearby, query_rent_nearby
from .predict import predict_price
from .rag import search_area_info
from .anomaly import detect_anomaly


def get_tools():
    return [
        query_trade_data,
        query_rent_data,
        query_trade_nearby,
        query_rent_nearby,
        predict_price,
        search_area_info,
        detect_anomaly,
    ]
