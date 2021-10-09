class xlsx:
    Features = []
    def __init__(self, filename):
        import pandas as pd    # Importing pandas
        self.file = pd.read_excel(filename, sheet_name='Sheet1')    # Read the file data

    def data(self):
    # Rearrange the datas in a list and return
        self.title = list(self.file)
        datas = []
        for i in self.title:
            datas.append(list(self.file[i]))

        return datas

    def titles(self):
        self.title = list(self.file)
        return (self.title)



class LinearRegression:
    X, Y, slopes, c = [], [], [], 0   #  Initialize the variables
    means, squares, XYs, meansOfXY, meansOfSquared, squaresOfMeans, yValues = [], [], [], [], [], [], []  # Temporary lists for calculation
    sumOfX, sumOfY, sumOfSquares, sumOfProducts = [], [], [], []

    data_fitted, predicted = False, False


    def fit(self, data):   #  Fit the data
        self.X = data[0]   #  Initialize X values
        self.Y = data[-1]   #  Initialize Y values
        self.data = data
        self.data_fitted = True


    def mul(self, x, y):   #  Multiply 2 lists
        multiplied = []
        if len(x)==len(y):
            for i in range(len(x)):
                multiplied.append(x[i]*y[i])
        return multiplied


    def square(self, x):   #  Square a list
        squared =[]
        for y in x:
            squared.append(y*y)
        return squared


    def mean(self, x):   #  Find mean of listed numbers
        # Mean of x, y, z is (x+y+z)/3
        mean_val = 0
        for i in x:
            mean_val += i
        return mean_val/len(x)


    def slope(self, x, y):   #  Find the slope
        #  m = [{mean(x)*mean(y) - mean(x*y)} / {mean(x^2) - mean(x^2)}]
        lob = (self.mean(x)*self.mean(y)) - (self.mean(self.mul(x, y)))
        hor = (self.mean(x)*self.mean(x)) - self.mean(self.square(x))
        return lob/hor


    def intercept(self):   #  Find the Y-Intercept
        #  c = mean(y) - m*mean(x)
        self.c = self.means[-1]
        for i in range(len(self.slopes)):
            self.c -= self.slopes[i] * self.means[i]

        print("Y-Intercept (α0) =", self.c)


    def initialize_data(self):
        self.yValues = self.data[-1]

        for i in self.data[:-1]:
            self.sumOfX.append(sum(i))
        self.sumOfY = sum(self.yValues)

        for i in self.data:
            self.means.append(self.mean(i))

        for i in self.data[:-1]:
            self.squares.append(self.square(i))

        for i in self.squares:
            self.meansOfSquared.append(self.mean(i))

        for i in self.data[:-1]:
            self.XYs.append(self.mul(i, self.data[-1]))

        for i in self.XYs:
            self.meansOfXY.append(self.mean(i))

        for i in self.squares:
            self.sumOfSquares.append(sum(i))

        for i in self.means[:-1]:
            self.squaresOfMeans.append(i*i)

        print("Means of [X1, X2, Y] =", self.means)
        print("Squared [X1, X2, Y] =", self.squares)
        print("Multiplications of xy [X1*Y, X2*Y] =", self.XYs)
        print("Means of the multiplications [mean of X1*Y, mean of X2*Y] =", self.meansOfXY)

    def slope_calc(self):
        lob, hor = [], []

        for i in range(len(self.data)-1):
            lob.append((self.means[i]*self.means[-1]) - self.meansOfXY[i])

        for i in range(len(self.data)-1):
            hor.append(self.squaresOfMeans[i] - self.meansOfSquared[i])

        for i in range(len(lob)):
            self.slopes.append(lob[i] / hor[i])

        print("Slopes [α1, α2] =", self.slopes)


    def predict(self, features):    #  Predict the Y values depending on X
                if self.data_fitted:
                        pass
                else:
                        raise Exception("DataNotFittedError : Call fit method before calling predict method in order to fit the training data to the model")

                self.initialize_data()
                self.slope_calc()
                self.intercept()
                slopes = self.slopes

                result = 0

                for i in range(len(features)):
                    result += (self.slopes[i]*features[i])

                self.predicted = True

                return result+self.c


    def r_squared(self):    # Calculate the r-squared value
                if self.predicted:    # If the predict methos isn't called, raise an error
                        pass
                else:
                        raise Exception("NotPredictedError : Call predict method first to calculate the r-squared value.")

                original_y = self.Y
                predicted_y = list(map(self.predict, self.X))    # Predict all Y values by our model
                lob, hor = 0, 0

                # Apply the formula
                for i in range(len(original_y)):
                        lob += (predicted_y[i]-self.mean(original_y))**2
                for i in range(len(original_y)):
                        hor += (original_y[i]-self.mean(original_y))**2
                        result = float("{:.4f}".format(lob/hor))
                return (result)


    def plot(self):
        if self.data_fitted:    # If data is not fitted, raise an error
            pass
        else:
            raise Exception("DataNotFittedError : Call fit method before calling plot method in order to plot the model")

        import matplotlib.pyplot as plt    # Import matplotlib for plotting and scattering

        plt.scatter(self.X, self.Y)    # Scatter the data points first
        predicted_y = list(map(self.predict, self.X))    # Predict all Y values by our model
        plt.plot(self.X, predicted_y)    # Plot the best fit line
        plt.show()    # Show the plot



if __name__ == '__main__':
    model = LinearRegression()  # Create the model

    model.fit([[60, 62, 67, 70, 71, 72, 75, 78], [22, 25, 24, 20, 15, 14, 14, 11], [140, 155, 159, 179, 192, 200, 212, 215]])  # Fit the data to our model
    #model.fit([[6, 8, 12, 14, 18], [350, 775, 1150, 1395, 1675]])
    
    print("\nX1 = 60, X2 = 22, Y =", model.predict([60, 22]), "(Predicted)")
    print("X1 = 60, X2 = 22, Y =", model.data[-1][0], "(Original)")
