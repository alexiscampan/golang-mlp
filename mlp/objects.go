package mlp

type Pattern struct {
	Features            []float64
	SinglRawExpectation string
	SingleExpectation   float64
	MultipleExpectation []float64
}

type NeuronUnit struct {
	Weights []float64
	Bias    float64
	Lrate   float64
	Value   float64
	Delta   float64
}

type NeuralLayer struct {
	NeuronUnits []NeuronUnit
	Length      int
}

type MultiLayerNetwork struct {
	L_rate       float64
	NeuralLayers []NeuralLayer
	T_func       transferFunction
	T_func_d     transferFunction
}
