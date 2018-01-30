package main

import (
	"fmt"
	"math"
	"math/rand"
)

type matrix [][]float64

//We shall create a neural net with 1 hidden layer and a single output neuron
type neuralNet struct {
	input           matrix
	prediction      matrix
	target          matrix
	errors          matrix
	learningRate    float64
	weightsHidden   matrix
	biasHidden      matrix
	weightsOutput   matrix
	biasOutput      matrix
	hiddenLayer     matrix
	activatedHidden matrix
}

func dotproduct(input []float64, weights []float64) float64 {
	sum := 0.0
	for i := 0; i < len(input); i++ {
		sum = sum + (input[i] * weights[i])
	}
	return sum
}

func returnCol(X matrix, column int) []float64 {
	rowNum := len(X)
	var col []float64
	for i := 0; i < rowNum; i++ {
		col = append(col, X[i][column])
	}
	return col
}

func transpose(X matrix) matrix {
	var t matrix
	for i := 0; i < len(X[0]); i++ {
		t = append(t, returnCol(X, i))
	}
	return t
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func derivativeSigmoid(x float64) float64 {
	return x * (1 - x)
}

func mtrxMult(A matrix, B matrix) matrix {
	colNum := len(A[0])
	//fmt.Println("colNum", colNum)
	rowNum := len(B)
	//fmt.Println("rowNum", rowNum)
	if colNum != rowNum {
		fmt.Println("Error, column, row dimensions do not match")
	}

	newRowNum := len(A)
	newMtrx := make(matrix, newRowNum)

	for i := 0; i < len(A); i++ {
		var row []float64
		for j := 0; j < len(B[0]); j++ {
			row = append(row, dotproduct(A[i], returnCol(B, j)))
		}
		newMtrx[i] = row
		//fmt.Println(row)
	}
	return newMtrx
}

func (n *neuralNet) initWeights(lr float64) {
	s1 := rand.NewSource(42)
	r1 := rand.New(s1)

	bH := make(matrix, 1, len(n.input))
	for i := 0; i < len(n.input); i++ {
		bH[0] = append(bH[0], r1.Float64())
	}
	bO := make(matrix, 1, 1)
	bO[0] = append(bO[0], r1.Float64())

	var wH matrix
	for i := 0; i < len(n.input[0]); i++ {
		var row []float64
		for j := 0; j < len(n.input); j++ {
			row = append(row, r1.Float64())
		}
		wH = append(wH, row)
	}

	var wO matrix
	for i := 0; i < len(n.input); i++ {
		wO = append(wO, []float64{r1.Float64()})
	}

	n.biasHidden = bH
	n.biasOutput = bO
	n.weightsHidden = wH
	n.weightsOutput = wO
	n.learningRate = lr
}

func (n *neuralNet) calcHiddenLayer() {
	m := mtrxMult(n.input, n.weightsHidden)
	b := n.biasHidden
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			m[i][j] = m[i][j] + b[0][j]
		}
	}
	n.hiddenLayer = m
}

func (n *neuralNet) activateHiddenLayer() {
	m := n.hiddenLayer
	for i := 0; i < len(n.hiddenLayer); i++ {
		for j := 0; j < len(n.hiddenLayer[0]); j++ {
			m[i][j] = sigmoid(m[i][j])
		}
	}
	n.activatedHidden = m
}

func (n *neuralNet) makePrediction() {
	m := mtrxMult(n.activatedHidden, n.weightsOutput)
	b := n.biasOutput
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			m[i][j] = m[i][j] + b[0][0]
			m[i][j] = sigmoid(m[i][j])
		}
	}

	n.prediction = m

}

func (n *neuralNet) calculateDerivatives() (matrix, matrix, matrix) {
	var Errors matrix
	for i := 0; i < len(n.target); i++ {
		e := (n.target[i][0] - n.prediction[i][0])
		Errors = append(Errors, []float64{e})
	}
	n.errors = Errors
	var predictionDerivative matrix
	for i := 0; i < len(n.prediction); i++ {
		d := derivativeSigmoid(n.prediction[i][0])
		predictionDerivative = append(predictionDerivative, []float64{d})
	}

	var HiddenDerivative matrix
	for i := 0; i < len(n.activatedHidden); i++ {
		var row []float64
		for j := 0; j < len(n.activatedHidden[0]); j++ {
			d := derivativeSigmoid(n.activatedHidden[i][j])
			row = append(row, d)
		}
		HiddenDerivative = append(HiddenDerivative, row)
	}
	return Errors, predictionDerivative, HiddenDerivative
}
func (n *neuralNet) findDeltas() (matrix, matrix) {
	e, p, h := n.calculateDerivatives()
	//fmt.Println("Derivative Hidden", h)
	var predictionDelta matrix
	for i := 0; i < len(e); i++ {
		d := e[i][0] * p[i][0]
		predictionDelta = append(predictionDelta, []float64{d})
	}

	errorHidden := mtrxMult(predictionDelta, transpose(n.weightsOutput))
	//fmt.Println("Error Hidden", errorHidden)

	var deltaHidden matrix
	for i := 0; i < len(h); i++ {
		var row []float64
		for j := 0; j < len(h[0]); j++ {
			d := h[i][j] * errorHidden[i][j]
			row = append(row, d)
		}
		deltaHidden = append(deltaHidden, row)
	}

	return predictionDelta, deltaHidden
}

func (n *neuralNet) updateWeights() {
	pD, hD := n.findDeltas()
	lr := n.learningRate
	m := mtrxMult(transpose(n.activatedHidden), pD)
	var newWeightsOut matrix
	for i := 0; i < len(pD); i++ {
		pr := lr*m[i][0] + n.weightsOutput[i][0]
		newWeightsOut = append(newWeightsOut, []float64{pr})
	}
	var newWeightsHidden matrix
	hm := mtrxMult(transpose(n.input), hD)
	for i := 0; i < len(hm); i++ {
		var row []float64
		for j := 0; j < len(hm[0]); j++ {
			row = append(row, lr*hm[i][j]+n.weightsHidden[i][j])
		}
		newWeightsHidden = append(newWeightsHidden, row)
	}

	var newBiasOut matrix
	sum := 0.0
	for i := 0; i < len(pD); i++ {
		sum += pD[i][0]
	}
	oldBiasOut := n.biasOutput[0][0]
	newBiasOut = append(newBiasOut, []float64{oldBiasOut + sum*lr})

	var newBiasHidden matrix
	for i := 0; i < len(hD); i++ {
		var row []float64
		sum := 0.0
		for j := 0; j < len(hD[0]); j++ {
			sum += hD[i][j]
			row = append(row, sum*lr)
		}
		newBiasHidden = append(newBiasHidden, row)
		newBiasHidden[0][i] += n.biasHidden[0][i]
	}

	n.weightsHidden = newWeightsHidden
	n.weightsOutput = newWeightsOut
	n.biasOutput = newBiasOut
	n.biasHidden = newBiasHidden
}

func (n *neuralNet) forwardPropagation() {
	n.calcHiddenLayer()
	n.activateHiddenLayer()
	n.makePrediction()
}

func (n *neuralNet) printNet() {
	fmt.Println("Input:", n.input)
	fmt.Println("___________")
	fmt.Println("Target:", n.target)
	fmt.Println("___________")
	fmt.Println("Prediction:", n.prediction)
	fmt.Println("___________")
	fmt.Println("Error:", n.errors)
}

func main() {
	X := matrix{
		{0, 1},
		{1, 1},
		{1, 0},
		{0, 0},
	}
	y := matrix{{1}, {0}, {1}, {0}} // Trying to predict

	n := neuralNet{
		input:  X,
		target: y,
	}

	n.initWeights(.1)

	for i := 0; i < 100000; i++ {
		n.forwardPropagation()
		n.updateWeights()
	}
	n.printNet()

}
