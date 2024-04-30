import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class Fuzzy:
    
    def __init__(self) -> None:
        self.consunto_pa()
        self.consunto_pulso()
        self.consunto_resp()
        self.consunto_gravidade()
        regras = self.define_regras()
        
        sif_ctrl = ctrl.ControlSystem(regras)
        self.sif = ctrl.ControlSystemSimulation(sif_ctrl)
        
    def consunto_pa(self):
        self.pa = ctrl.Antecedent(np.arange(-10, 10.1, 0.1), 'pa')
        
        self.pa['max'] = fuzz.trimf(self.pa.universe, [-8, -0, 8])
        self.pa['baixa'] = fuzz.trimf(self.pa.universe, [-10, -10, -5])
        self.pa['alta'] = fuzz.trimf(self.pa.universe, [5, 10, 10])
        
    def consunto_pulso(self):
        self.pulso = ctrl.Antecedent(np.arange(0, 201, 1), 'pulso')
        
        self.pulso['baixa'] = fuzz.trimf(self.pulso.universe, [0, 0, 75])
        self.pulso['normal'] = fuzz.trimf(self.pulso.universe, [30, 80, 140])
        self.pulso['alta'] = fuzz.trimf(self.pulso.universe, [110, 200, 200])
        
    def consunto_resp(self):
        self.resp = ctrl.Antecedent(np.arange(0, 201, 1), 'resp')
        
        self.resp['baixa'] = fuzz.trapmf(self.resp.universe, [0, 0, 6.5 , 12])
        self.resp['normal'] = fuzz.trimf(self.resp.universe, [10, 15, 20])
        self.resp['alta'] = fuzz.trapmf(self.resp.universe, [17, 21, 22, 22])
        
    def consunto_gravidade(self):
        self.gravidade = ctrl.Consequent(np.arange(0, 201, 1), 'gravidade')
        
        self.gravidade['Crítico'] = fuzz.trapmf(self.gravidade.universe, [0, 0, 20, 28])
        self.gravidade['Instável'] = fuzz.trapmf(self.gravidade.universe, [25, 32, 45, 53])
        self.gravidade['Pot. Estável'] = fuzz.trapmf(self.gravidade.universe, [48, 52, 66, 78])
        self.gravidade['Estável'] = fuzz.trapmf(self.gravidade.universe, [72, 76, 100, 100])
        
    def define_regras(self):
        regras = [ ]
    
        regras.append(ctrl.Rule(self.pa['max'] & self.pulso['normal'] & self.resp['normal'], self.gravidade['Estável']))
        regras.append(ctrl.Rule(self.pa['baixa'] & self.pulso['baixa'] & self.resp['baixa'], self.gravidade['Crítico']))
        regras.append(ctrl.Rule(self.pa['baixa'] & self.pulso['baixa'] & self.resp['normal'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['baixa'] & self.pulso['baixa'] & self.resp['alta'], self.gravidade['Crítico']))
        regras.append(ctrl.Rule(self.pa['baixa'] & self.pulso['normal'] & self.resp['baixa'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['baixa'] & self.pulso['normal'] & self.resp['normal'], self.gravidade['Pot. Estável']))
        regras.append(ctrl.Rule(self.pa['baixa'] & self.pulso['normal'] & self.resp['alta'], self.gravidade['Pot. Estável']))
        regras.append(ctrl.Rule(self.pa['baixa'] & self.pulso['alta'] & self.resp['baixa'], self.gravidade['Crítico']))
        regras.append(ctrl.Rule(self.pa['baixa'] & self.pulso['alta'] & self.resp['normal'], self.gravidade['Pot. Estável']))
        regras.append(ctrl.Rule(self.pa['baixa'] & self.pulso['alta'] & self.resp['alta'], self.gravidade['Crítico']))

        regras.append(ctrl.Rule(self.pa['max'] & self.pulso['baixa'] & self.resp['baixa'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['max'] & self.pulso['baixa'] & self.resp['normal'], self.gravidade['Pot. Estável']))
        regras.append(ctrl.Rule(self.pa['max'] & self.pulso['baixa'] & self.resp['alta'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['max'] & self.pulso['normal'] & self.resp['baixa'], self.gravidade['Pot. Estável']))
        regras.append(ctrl.Rule(self.pa['max'] & self.pulso['normal'] & self.resp['alta'], self.gravidade['Pot. Estável']))
        regras.append(ctrl.Rule(self.pa['max'] & self.pulso['alta'] & self.resp['baixa'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['max'] & self.pulso['alta'] & self.resp['normal'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['max'] & self.pulso['alta'] & self.resp['alta'], self.gravidade['Instável']))

        regras.append(ctrl.Rule(self.pa['alta'] & self.pulso['baixa'] & self.resp['baixa'], self.gravidade['Crítico']))
        regras.append(ctrl.Rule(self.pa['alta'] & self.pulso['baixa'] & self.resp['normal'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['alta'] & self.pulso['baixa'] & self.resp['alta'], self.gravidade['Crítico']))
        regras.append(ctrl.Rule(self.pa['alta'] & self.pulso['normal'] & self.resp['baixa'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['alta'] & self.pulso['normal'] & self.resp['normal'], self.gravidade['Pot. Estável']))
        regras.append(ctrl.Rule(self.pa['alta'] & self.pulso['normal'] & self.resp['alta'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['alta'] & self.pulso['alta'] & self.resp['baixa'], self.gravidade['Crítico']))
        regras.append(ctrl.Rule(self.pa['alta'] & self.pulso['alta'] & self.resp['normal'], self.gravidade['Instável']))
        regras.append(ctrl.Rule(self.pa['alta'] & self.pulso['alta'] & self.resp['alta'], self.gravidade['Crítico']))
        
        return regras
    
    def compute(self, pa, pulso, resp):
        self.sif.input['pa'] = pa
        self.sif.input['pulso'] = pulso
        self.sif.input['resp'] = resp
        
        self.sif.compute()
        
        output = self.sif.output['gravidade']
        termo_saida = max(self.gravidade.terms.keys(), key=lambda term: fuzz.interp_membership(self.gravidade.universe, self.gravidade[term].mf, output))
        
        if termo_saida == 'Crítico':
            return 1
        elif termo_saida == 'Instável':
            return 2
        elif termo_saida == 'Pot. Estável':
            return 3
        else:
            return 4
