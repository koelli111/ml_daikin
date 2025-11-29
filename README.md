# Online-RLS (oppiva) ohjain Daikin demand-selectille.
# Ominaisuudet:
# - min 30 %, max 100 %
# - 6h ulkoennusteen keskiarvo feed-forwardina
# - pakotettu suunta-askel (aina vähintään step_limit kohti setpointtia deadbandin ulkopuolella)
# - deadband, step-limit, monotonisuus ja selectin pykälöinti
# - oppiminen (RLS) jäädytetään defrostin aikana (sensor.faikin_liquid < 20)
# - parametrisäilö input_text.daikin_rls_params (vain theta tallennetaan, ei P)
