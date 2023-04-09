package mlp

import (
	log "github.com/sirupsen/logrus"
)

func PrepareLayer(n int, p int) (l NeuralLayer) {
	l = NeuralLayer{NeuronUnits: make([]NeuronUnit, n), Length: n}
	for i := 0; i < n; i++ {
		RandomNeuronInit(&l.NeuronUnits[i], p)
	}
	log.WithFields(log.Fields{
		"level":               "info",
		"msg":                 "multilayer perceptron init completed",
		"neurons":             len(l.NeuronUnits),
		"lengthPreviousLayer": l.Length,
	}).Info("Complete NeuralLayer init")
	return
}

func PrepareMLPNet(l []int, lr float64, tf, trd transferFunction) (mlp MultiLayerNetwork) {
	mlp.L_rate = lr
	mlp.T_func = tf
	mlp.T_func_d = trd

	mlp.NeuralLayers = make([]NeuralLayer, len(l))

	for idx, ql := range l {
		if idx != 0 {
			mlp.NeuralLayers[idx] = PrepareLayer(ql, l[idx-1])
		} else {
			mlp.NeuralLayers[idx] = PrepareLayer(ql, 0)
		}
	}
	log.WithFields(log.Fields{
		"level":          "info",
		"msg":            "multilayer perceptron init completed",
		"layers":         len(mlp.NeuralLayers),
		"learningRate: ": mlp.L_rate,
	}).Info("Complete Multilayer Perceptron init.")
	return
}
