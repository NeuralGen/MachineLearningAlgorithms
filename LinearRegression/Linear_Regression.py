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

        

class LinearRegression: 
    X, Y, slopes, c = [], [], [], 0   #  Initialize the variables
    data_fitted, predicted = False, False


    def fit(self, data):   #  Fit the data
        self.X = data[0]   #  Initialize X values
        self.Y = data[1]   #  Initialize Y values
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

    
    def intercept(self, *args):   #  Find the Y-Intercept
        #  c = mean(y) - m*mean(x)
        args = list(args)
        y = args[-1]   #  Initialize y value
        intercept = self.mean(y)
        del args[-1]
        for i in range(len(args)):
            intercept -= self.slope(args[i], y)*self.mean(args[i])
        return intercept


    def slope_calc(self):   #  Calculate the needed slopes
        for i in range(len(self.X)):
            self.slopes.append(self.slope(self.X[i], self.Y))

        
    def predict(self, x):    #  Predict the Y values depending on X
                if self.data_fitted:
                        pass
                else:
                        raise Exception("DataNotFittedError : Call fit method before calling predict method in order to fit the training data to the model")
                
                m = self.slope(self.X, self.Y)   #  Calculate the slopes
                c = self.intercept(self.X, self.Y)   #  Calculate the intercepts
                self.predicted = True
                return (m*x)+c

        
    def r_squared(self):    # Calculate the r-squared value
                if self.predicted:    # If the predict methos isn't called, raise an error
                        pass
                else:
                        raise Exception("NotPredictedError : Call predict method first to calculate the r-squared value.")
                
                original_y = self.Y
                predicted_y = list(map(self.predict, self.X))    # Predict all Y values by our model
                lob, hor = 0, 0

                # Apply the formula
                for i in range(len(original_y))
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

    data = xlsx('data.xlsx')  # Read the data from data.xlsx
    model.fit(data.data())  # Fit the data to our model
    print(model.predict(10))  # Print the price of 10 inch
    model.plot()  # Plot the model into the graph
