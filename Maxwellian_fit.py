import numpy as np

def Maxwellian_fit(Emin,Emax,counts,energy):
    # Performing LinearRegression
    from sklearn.model_selection import train_test_split
    binsize = energy[1]-energy[0]
    learn_min = int(Emin/binsize)
    learn_max = int(Emax/binsize)
    X = energy[learn_min:learn_max].reshape((-1, 1))
    y = np.log10(counts[learn_min:learn_max])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)
    print('intercept:',regressor.intercept_)
    print('gradient:',regressor.coef_[0])
    print('R-squared:',regressor.score(X,y))
    # Y = C * exp(-E/scale) ; scale = kT [MeV]
    # log10 Y = log10 C - (E/scale) * log10 e; regressor.coef_[0] = -log10 e/scale
    Temperature = round(-np.log10(2.71828) / regressor.coef_[0],1)
    print('Temperature [MeV]:',Temperature)
    return Temperature

