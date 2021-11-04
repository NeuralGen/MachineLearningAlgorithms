class KNN:
    Distances = []
    def train(self, *args):
        self.Features = list(args)
        self.X1 = self.Features[0]
        self.X2 = self.Features[1]
        self.Y = self.Features[2]
        

    def set_K(self):
        sqrt = round(len(self.X1)**(1/2))
        if sqrt%2 == 0: self.K = int(sqrt+1)
        else: self.K = int(sqrt)


    def EuclideanDistance(self, point1, point2):
        return ( (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 ) ** (1/2)


    def GetDistances(self):
        for i in range(len(self.X1)):
            distance = self.EuclideanDistance([self.X1[i], self.X2[i]], [self.UnknownValue[0], self.UnknownValue[1]])
            self.Distances.append([distance, self.Y[i]])
        self.Distances.sort()


    def most_frequent(self, List):
        counter = 0
        num = List[0]
         
        for i in List:
            curr_frequency = List.count(i)
            if(curr_frequency > counter):
                counter = curr_frequency
                num = i
        return num


    def predict(self, *args):
        self.UnknownValue = list(args)
        self.set_K()
        self.GetDistances()
        dataPoints = []
        classes = []
        for i in self.Distances:
            dataPoints.append(i[-1])

        for Class in range(self.K):
            classes.append(dataPoints[Class])
        return (self.most_frequent(classes))
    

    def plot(self):
        import matplotlib.pyplot as plt
        #predicted_y = list(map(self.predict, self.X))
        plt.scatter(self.X1, self.X2)
        #plt.scatter()
        plt.show()



Height, Weight, Class = [4.2,4.0,3.8,2.0,2.7,1.7,2.7,1.2,2.2,0.3], [2.8,2.0,0.5,1.5,2.5,3.2,4.0,5.2,6.2,6.2], ["Dog","Dog","Dog","Dog","Dog","Cat","Cat","Cat","Cat","Cat"]
knn = KNN()
knn.train(Height, Weight, Class)
print(knn.predict(2.2,3))
knn.plot()
#print(Distances)
