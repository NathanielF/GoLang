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

///////// Rounding function for readibility
func round(val float64, roundOn float64, places int) (newVal float64) {
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

//////////// Classificiation function based of Boolean
func binary(x bool) float64 {
	if x {
		return -1.0
	}
	return 1.0
}

//////////////// Function for updating the weights based on observed deltas and learning rate

func update(input []float64, delta float64, weights []float64, rate float64) []float64 {
	for i := 1; i < len(weights); i++ {
		//fmt.Println("Input", weights[i], "delta", delta, "rate", rate, "factor", input[i])
		weights[i] = round(weights[i], .5, 3) + delta*rate*input[i]
		//fmt.Println(weights[i])
	}
	//fmt.Println(weights[0], delta, rate)
	weights[0] = round(weights[0], .5, 3) + delta*rate
	return weights
}

///////////////// Dot product function
func dotproduct(input []float64, weights []float64) float64 {
	sum := 0.0
	for i := 0; i < len(input); i++ {
		sum = sum + (input[i] * weights[i])
	}
	return sum
}

///////// Heaviside activation function
func heaviside(x float64) float64 {
	if x >= 0.0 {
		return 1.0
	}
	return -1.0
}

//////////// Prediction function combining dotproduct and heaviside
func predict(input []float64, weights []float64) float64 {
	sum := dotproduct(input, weights)
	return heaviside(sum)
}

//////////// Process function looping over each observation row-vector and updating the weights
///////////  Returns new weights and error count for each prediction based on the observed values.
func process(inputs [][]float64, weights []float64, rate float64) ([]float64, int) {
	errorcount := 0
	for i := 0; i <= len(inputs)-1; i++ {
		//fmt.Println("Input", i, ":", inputs[i])
		target := inputs[i][len(inputs[i])-1]
		//fmt.Println("Target", i, ":", target)
		prediction := predict(inputs[i][0:len(inputs[i])-1], weights)
		//dotprod := dotproduct(inputs[i][0:len(inputs[i])-1], weights)
		//fmt.Println("Prediction", i, ":", prediction)
		delta := target - prediction
		//fmt.Println("Target", i, ":", inputs[i][len(inputs[i])-1], "Delta", delta, "Dotproduct", dotprod, "Prediction", predict(inputs[i][0:len(inputs[i])-2], weights))
		weights = update(inputs[i], delta, weights, rate)
		//fmt.Println("Round", i, "inputs", inputs[i], ":", "Weights", weights, "dot", dotprod, "target", target)
		if delta != 0.0 {
			errorcount = errorcount + 1
		}
	}
	return weights, errorcount
}

func perceptron(inputs [][]float64, weights []float64, rate float64, iterations int) ([]float64, []int) {
	var finalweights []float64
	var finalerrors []int
	var finalerror int
	for i := 0; i <= iterations; i++ {
		fmt.Println("Input weight loop", i, ":", finalweights)
		finalweights, finalerror = process(inputs, weights, rate)
		finalerrors = append(finalerrors, finalerror)
		//fmt.Println(finalweights, finalerrors)
	}
	return finalweights, finalerrors
}

/////////// We read the file and pull out each row as an element of a slice.
////////// Each element of the slice is a string like: "5.9,3.0,5.1,1.8,Iris-virginica"

func main() {
	////////Loading in the iris data as a slice of row strings
	iris, err := readLines("/Users/nathanielforde/Desktop/Data Science/Mathematical Probability/iris.data.csv")
	if err != nil {
		return
	}

	str := iris
	//fmt.Println(str[1])

	////////////   We now extract the various row elements initially as strings,
	///////////    from the row-string ( e.g from  "5.9,3.0,5.1,1.8,Iris-virginica"), we extrace "5.9" and convert
	////////////   to force the values into numeric format. We then reconstruct the row
	////////////   and  put the details into a slice of slices called
	///////////   inputs. We only need the first hundred observations of iris data. We will use a perceptron to
	///////////   distinguish the Iris-virginica and Iris-setosa based on Sepal and Petal length.

	inputs := make([][]float64, 100) ////// Declaring the empty slice of slices. Our design matrix.

	for i := 0; i < 100; i++ { /////// pull out and convert each substring into a number
		var input []float64
		p, _ := strconv.ParseFloat(strings.Split(string(str[i]), ",")[0], 64) ///// Sepal Length
		//q, _ := strconv.ParseFloat(strings.Split(string(str[i]), ",")[1], 64)
		r, _ := strconv.ParseFloat(strings.Split(string(str[i]), ",")[2], 64) ///// Petal Length
		//s, _ := strconv.ParseFloat(strings.Split(string(str[i]), ",")[3], 64)
		t := binary(strings.Split(string(str[i]), ",")[4] == "Iris-setosa")
		input = append(input, 1.0) /////// Recreating our row vectors for each observation
		input = append(input, p)
		//input = append(input, q)
		input = append(input, r)
		//input = append(input, s)
		input = append(input, t)
		inputs[i] = input /////// Defining each row-vector observation (e.g [1, 5.9, 5.1, -1])
		////// as an instance of our design matrix  ["intercept column", "sepal length", "petal length", target]
	}

	var w = []float64{0, 0, 0} ///////// initialising our weights

	fmt.Println(perceptron(inputs, w, .1, 10))

}
