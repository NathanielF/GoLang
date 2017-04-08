package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// readLines reads a whole file into memory
// and returns a slice of its lines.
func readLines(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}

func Round(val float64, roundOn float64, places int) (newVal float64) {
	var round float64
	pow := math.Pow(10, float64(places))
	digit := pow * val
	_, div := math.Modf(digit)
	if div >= roundOn {
		round = math.Ceil(digit)
	} else {
		round = math.Floor(digit)
	}
	newVal = round / pow
	return
}

func binary(x bool) float64 {
	if x {
		return -1.0
	}
	return 1.0
}

func update(input []float64, delta float64, weights []float64, rate float64) []float64 {
	for i := 1; i < len(weights); i++ {
		//fmt.Println("Input", weights[i], "delta", delta, "rate", rate, "factor", input[i])
		weights[i] = Round(weights[i], .5, 3) + delta*rate*input[i]
		//fmt.Println(weights[i])
	}
	//fmt.Println(weights[0], delta, rate)
	weights[0] = Round(weights[0], .5, 3) + delta*rate
	return weights
}

func dotproduct(input []float64, weights []float64) float64 {
	sum := 0.0
	for i := 0; i < len(input); i++ {
		sum = sum + (input[i] * weights[i])
	}
	return sum
}

func predict(input []float64, weights []float64) float64 {
	sum := 0.0
	for i := 0; i < len(input); i++ {
		sum = sum + (input[i] * weights[i])
	}
	return heaviside(sum)
}

func heaviside(x float64) float64 {
	if x >= 0.0 {
		return 1.0
	}
	return -1.0
}

func sumvector(x []float64) float64 {
	sum := 0.0
	for i := 0; i <= len(x); i++ {
		fmt.Println(x[i])
		if x[i] == 1.0 {
			sum = sum + 1
		}
	}
	return sum
}

func process(inputs [][]float64, weights []float64, rate float64) ([]float64, int) {
	errorcount := 0
	fmt.Println(weights)
	for i := 0; i <= len(inputs)-1; i++ {
		//fmt.Println("Input", i, ":", inputs[i])
		target := inputs[i][len(inputs[i])-1]
		//fmt.Println("Target", i, ":", target)
		prediction := predict(inputs[i][0:len(inputs[i])-1], weights)
		dotprod := dotproduct(inputs[i][0:len(inputs[i])-1], weights)
		//fmt.Println("Prediction", i, ":", prediction)
		delta := target - prediction
		//fmt.Println("Target", i, ":", inputs[i][len(inputs[i])-1], "Delta", delta, "Dotproduct", dotprod, "Prediction", predict(inputs[i][0:len(inputs[i])-2], weights))
		weights = update(inputs[i], delta, weights, rate)
		fmt.Println("Round", i, "inputs", inputs[i], ":", "Weights", weights, "dot", dotprod, "target", target)
		if delta != 0.0 {
			errorcount = errorcount + 1
		}
	}
	return weights, errorcount
}

func perceptron(inputs [][]float64, weights []float64, rate float64, iterations int) ([]float64, int) {
	var finalweights []float64
	var finalerrors int
	for i := 0; i <= iterations; i++ {
		finalweights, finalerrors = process(inputs, weights, rate)
		//fmt.Println(finalweights, finalerrors)
	}
	return finalweights, finalerrors
}

/////////// We read the file and pull out each row as an element of a slice.
////////// Each element of the slice is a string like: "5.9,3.0,5.1,1.8,Iris-virginica"

func main() {

	iris, err := readLines("/Users/nathanielforde/Desktop/Data Science/Mathematical Probability/iris.data.csv")
	if err != nil {
		return
	}

	str := iris
	////////////   We now extract the various row elements initially as strings, from the row-string
	////////////  to force the values into numeric format.
	inputs := make([][]float64, 100)
	//var target []float64

	for i := 0; i < 100; i++ {
		var input []float64
		p, _ := strconv.ParseFloat(strings.Split(string(str[i]), ",")[0], 64)
		//q, _ := strconv.ParseFloat(strings.Split(string(str[i]), ",")[1], 64)
		r, _ := strconv.ParseFloat(strings.Split(string(str[i]), ",")[2], 64)
		//s, _ := strconv.ParseFloat(strings.Split(string(str[i]), ",")[3], 64)
		t := binary(strings.Split(string(str[i]), ",")[4] == "Iris-setosa")
		input = append(input, 1.0)
		input = append(input, p)
		//input = append(input, q)
		input = append(input, r)
		//input = append(input, s)
		input = append(input, t)
		inputs[i] = input
	}
	var a = []float64{1, -0.4, 0.3, 1}
	var b = []float64{1, -0.3, -0.1, 1}
	var c = []float64{1, -0.2, 0.4, 1}
	var d = []float64{1, -0.1, 0.1, 1}
	var e = []float64{1, 0.1, -0.5, -1}
	var f = []float64{1, 0.2, -0.9, -1}
	var g = []float64{-0.2, -0.26, 1.04}
	var h = []float64{1, 4.6, 1.5}
	var w = []float64{0, 0, 0}
	//var c float64 = .1
	//	var d float64 = .3
	multi := make([][]float64, 8)
	multi[0] = a
	multi[1] = b
	multi[2] = c
	multi[3] = d
	multi[4] = e
	multi[5] = f
	multi[6] = g
	multi[7] = h

	//fmt.Println(multi[1][len(w)+1] - heaviside(multi[1][0:len(w)], w))
	//fmt.Println(dotproduct(a[0:len(w)], w))
	//fmt.Println(heaviside(multi[1][0:len(w)], w))
	//fmt.Println(multi[1][0:len(w)])
	//fmt.Println(update(multi[1], 2, update(multi[0], -1, w, .1), .1))
	//fmt.Println(len(str))
	//fmt.Println(update(multi[1], 0.0, w, .01))
	//fmt.Println(process(inputs, w, .01))
	fmt.Println(perceptron(inputs, w, .1, 10))
	//fmt.Println(multi[1][0:len(multi[1])-2], w)
	//fmt.Println(inputs[3][0 : len(inputs[3])-1])
	//fmt.Println(dotproduct(inputs[3][0:len(inputs[3])-1], g))
}
