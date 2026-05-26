from sqlalchemy import Column, Integer, Float, String, Date
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """아파트 매매 실거래 테이블 (realestate.db 스키마 기준)"""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    apt_name = Column(String, nullable=False)
    district = Column(String, nullable=False)       # 시군구명
    sgg_code = Column(String(5))                    # 시군구 코드
    area_exclusive = Column(Float)                  # 전용면적 (㎡)
    floor = Column(Integer)
    deal_amount = Column(Float)                     # 거래금액 (만원)
    deal_date = Column(String)                      # YYYY-MM-DD
    dealing_type = Column(String)                   # 중개거래 / 직거래
    build_year = Column(Integer)
    latitude = Column(Float)
    longitude = Column(Float)
