# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_classif
import re
import warnings
warnings.filterwarnings("ignore")

# %%


def custom_print(text, ansi_off=True):
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    ''', re.VERBOSE)

    text_output = text
    if(ansi_off):
        text_output = ansi_escape.sub('', str(text))
    print(text_output)


class FeatureSelection:
    X = []
    y = []
    X_train = X_test = y_train = y_test = None
    estimator = LogisticRegression(solver='lbfgs', max_iter=200)
    k = 5

    rfe_selector = None
    ch2_selector = None
    univariate_selector = None
    selector = None

    def __init__(self, ds_name, k):
        self.X, self.y = self.get_dataset(ds_name)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2)
        self.k = k
        rfe_selector = RFE(estimator=self.estimator,
                           n_features_to_select=self.k)
        self.rfe_selector = rfe_selector.fit(self.X_train, self.y_train)

        chi2_selector = SelectKBest(chi2, k=k)
        self.ch2_selector = chi2_selector.fit(self.X_train, self.y_train)

        univariate_selector = SelectKBest(f_classif, k=k)
        self.univariate_selector = univariate_selector.fit(
            self.X_train, self.y_train)

        self.selector = None

    def get_dataset(self, name):
        dataset = np.genfromtxt(
            f'./datasets/{name}.txt', delimiter=' ', skip_header=0)
        X = dataset[:, 1:]
        y = dataset[:, 0]

        return X, y

    def set_selector(self, name):
        if(name == "chi-square"):
            self.selector = self.ch2_selector
        elif(name == "rfe"):
            self.selector = self.rfe_selector
        elif(name == "univariate"):
            self.selector = self.univariate_selector
        else:
            self.selector = self.univariate_selector

    def get_selected_features_name(self, selector_name):
        self.set_selector(selector_name)
        return self.selector.get_feature_names_out()

    def original_report(self):
        clf = self.estimator.fit(self.X_train, self.y_train)
        train_preditions = clf.predict(self.X_train)
        test_preditions = clf.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, train_preditions)
        test_accuracy = accuracy_score(self.y_test, test_preditions)

        return {"accuracy": train_accuracy}, {"accuracy": test_accuracy}

    def selected_features_report(self, selector_name):
        self.set_selector(selector_name)
        cols = self.selector.get_support(indices=True)
        X_train_selected = self.X_train[:, cols]
        X_test_selected = self.X_test[:, cols]
        clf = self.estimator.fit(X_train_selected, self.y_train)
        train_preditions = clf.predict(X_train_selected)
        test_preditions = clf.predict(X_test_selected)
        train_accuracy = accuracy_score(self.y_train, train_preditions)
        test_accuracy = accuracy_score(self.y_test, test_preditions)

        return {"accuracy": train_accuracy}, {"accuracy": test_accuracy}

    def compare_metrics(self, previous, current):
        changed_by = 0
        if current == previous:
            return "has \x1b[33mnot changed\x1b[0m."
        try:
            changed_by = ((current - previous) / previous) * 100.0
            if(changed_by > 0):
                return f"has \x1b[92mincreased\x1b[0m by \x1b[92m{round(abs(changed_by), 2)}%\x1b[0m."
            else:
                return f"has \x1b[91mdecreased\x1b[0m by \x1b[91m{round(abs(changed_by), 2)}%\x1b[0m."
        except ZeroDivisionError:
            changed_by = 0
            return

    def report_results(self, ansi_off=True):
        original_train_reports, original_test_reports = self.original_report()
        chi2_train_reports, chi2_test_reports = self.selected_features_report(
            "chi-square")
        rfe_train_reports, rfe_test_reports = self.selected_features_report(
            "rfe")
        univariate_train_reports, univariate_test_reports = self.selected_features_report(
            "univariate")

        print("Chi-square:")
        print(
            f"{self.k} selected features: {self.get_selected_features_name('chi-square')}")
        changed_by = self.compare_metrics(
            original_train_reports["accuracy"], chi2_train_reports["accuracy"])
        custom_print(f"The accuracy on train dataset {changed_by}", ansi_off)
        changed_by = self.compare_metrics(
            original_test_reports["accuracy"], chi2_test_reports["accuracy"])
        custom_print(f"The accuracy on test dataset {changed_by}"), ansi_off
        print("RFE:")
        print(f"{self.k} selected features: {self.get_selected_features_name('rfe')}")
        changed_by = self.compare_metrics(
            original_train_reports["accuracy"], rfe_train_reports["accuracy"])
        custom_print(f"The accuracy on train dataset  {changed_by}", ansi_off)
        changed_by = self.compare_metrics(
            original_test_reports["accuracy"], rfe_test_reports["accuracy"])
        custom_print(f"The accuracy on test dataset {changed_by}", ansi_off)
        print("Univariate:")
        print(
            f"{self.k} selected features: {self.get_selected_features_name('univariate')}")
        changed_by = self.compare_metrics(
            original_train_reports["accuracy"], univariate_train_reports["accuracy"])
        custom_print(f"The accuracy on train dataset  {changed_by}", ansi_off)
        changed_by = self.compare_metrics(
            original_test_reports["accuracy"], univariate_test_reports["accuracy"])
        custom_print(f"The accuracy on test dataset {changed_by}", ansi_off)


# %%
all_datsets_name = ['anneal', 'diabetes', 'hepatitis', 'kr-vs-kp', 'vote']
k_s = [5, 10]

for ds_name in all_datsets_name:
    print('-'*25 + f' Dataset: {ds_name} ' + '-'*25)
    for k in k_s:
        custom_print(f"Examining with \x1b[96mk = {k}\x1b[0m")
        FS = FeatureSelection(ds_name, k)
        FS.report_results()
        print()

# %%
