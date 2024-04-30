import pickle


class DecisionTree():
    def __init__(self):
        with open('decision_tree_model.pkl', 'rb') as file:
            self.loaded_model = pickle.load(file)

    def compute(self, pa, pulso, resp):
        prediction = self.loaded_model.predict([[pa, pulso, resp]])
        
        return prediction[0]