from dataclasses import dataclass


@dataclass
class TrainDataModel:
    id: int

    an_e_w_rou: float  # Antera 3D 좌측 눈가 평균거칠기 측정값
    an_e_w_dep: float  # Antera 3D 좌측 눈가 주름 최대 깊이 측정값
    an_f_w_rou: float  # Antera 3D 미간 평균거칠기 측정값
    an_f_w_dep: float  # Antera 3D 미간 주름 최대 깊이 측정값
    cor_c_a: float  # Corneometer 좌측볼 수분 측정값
    cor_f_a: float  # Corneometer 미간 수분 측정값
    seb_c_o: float  # Sebumeter 좌측볼 유분 측정값
    seb_f_o: float  # Sebumeter 미간 유분 측정값
    mex_m: float  # Mexameter 색소부위 멜라닌 측정값
    mex_nm: float  # Mexameter 비색소부위 멜라닌 측정값
