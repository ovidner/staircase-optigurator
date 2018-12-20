import numpy as np
from openmdao.api import CaseReader

from optigurator.utils import recording_filename


def get_case_reader(problem_constants):
    return CaseReader(recording_filename(problem_constants.id))


def generate_valid_points(problem_constants, crm):
    for (i, case_id) in enumerate(crm.list_cases()):
        model_case = crm.get_case(case_id)

        if (
            model_case.outputs["usability.min_max_step_height"][1]
            <= problem_constants.step_height.upper
            and model_case.outputs["usability.min_max_step_depth"][0]
            >= problem_constants.step_depth.lower
            and model_case.outputs["usability.min_free_height"][0]
            > problem_constants.free_height_lower
        ):
            yield [
                model_case.outputs["price_availability.total_price"][0],
                model_case.outputs["usability.max_step_comfort_rule_deviation"][0],
                model_case.outputs["price_availability.total_delivery_time"][0],
                i,
            ]


def calculate(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    normalizedRowNr = 0

    # skapar en kopia på matrisen som normaliseras senare
    normalizedPoints = np.array(inputPoints.copy())

    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0

    for i in range(0, len(normalizedPoints)):
        # summerar värden kolonnvis till nämnare för normalisering
        sum1 = sum1 + normalizedPoints[i, 0] ** 2
        sum2 = sum2 + normalizedPoints[i, 1] ** 2
        sum3 = sum3 + normalizedPoints[i, 2] ** 2

    # definerar en vektor med normaliseringsvärden
    myarray_normalize = [sum1 ** 0.5, sum2 ** 0.5, sum3 ** 0.5, 1]

    # Normaliserar matrisen
    normalizedPoints = np.array(inputPoints) / np.array(myarray_normalize)

    while True:
        candidateRow = inputPoints[candidateRowNr]
        normalized = normalizedPoints[normalizedRowNr]
        normalizedPoints = np.delete(normalizedPoints, normalizedRowNr, 0)
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(normalizedPoints) != 0 and rowNr < len(normalizedPoints):
            row = normalizedPoints[rowNr]
            rowIP = inputPoints[rowNr]

            if dominates(
                row, normalized
            ):  # Går in om candidateRow är bättre än utmanaren.
                normalizedPoints = np.delete(normalizedPoints, rowNr, 0)
                inputPoints.remove(rowIP)
                dominatedPoints.add(tuple(rowIP))

            elif dominates(
                normalized, row
            ):  # Går in om utmanare är större än kandidaten.
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:  # Lägg till nondominated punkter till pareto
            ID = int(normalized[3])
            paretoPoints.add(tuple(candidateRow))

        if len(normalizedPoints) == 0:  # SLutar när man gått igenom alla punkter.
            break

    dp = np.array(list(dominatedPoints))
    pp = np.array(list(paretoPoints))

    return paretoPoints, dominatedPoints, dp, pp


def dominates(row, normalized):  # Beräknar om utmanare är bättre än candidate.
    return sum([row[x] >= normalized[x] for x in range(len(row) - 1)]) == len(row) - 1


def WeightPPpoints(pp, my_weights):

    Pareto_points = pp
    np.size(Pareto_points)
    Nrofrows_pareto = np.size(Pareto_points, 0)

    # skapar en vektor med ID
    ID_vektor = np.delete(Pareto_points, [0, 1, 2], 1).tolist()

    # skapar matris med outputvärden utan ID kolonn
    A = np.delete(Pareto_points, 3, 1)

    np.size(A)
    # definerar storleken på matrisen som kommer som paretopoints output
    Nrofcolumns = np.size(A, 1)
    Nrofrows = np.size(A, 0)
    sizeofA = (Nrofrows, Nrofcolumns)

    # Skapar matris som sedan fylls med bästa lösningarnas ID
    IDpoints = []

    # skapar en kopia på matrisen som normaliseras senare
    B = A.copy()
    sum1 = 0
    sum2 = 0
    sum3 = 0

    for i in range(0, Nrofrows):
        # summerar värden kolonnvis till nämnare för normalisering
        sum1 = sum1 + A[i, 0] ** 2
        sum2 = sum2 + A[i, 1] ** 2
        sum3 = sum3 + A[i, 2] ** 2

    # definerar en vektor med normaliseringsvärden
    myarray_normalize = [sum1 ** 0.5, sum2 ** 0.5, sum3 ** 0.5]

    # Normaliserar matrisen
    B = A / myarray_normalize

    # kopierar matrisen och multiplicerar kolonnvis med viktningar

    C = B.copy()

    # Loop för 5 olika viktningar -> 5 optimala pareto punkter som output

    for j in range(0, len(my_weights)):
        for i in range(0, Nrofrows):
            C[i, 0] = B[i, 0] * my_weights[j, 0]
            C[i, 1] = B[i, 1] * my_weights[j, 1]
            C[i, 2] = B[i, 2] * my_weights[j, 2]

        # Definera ideala värden A_positive samt icke ideala värden A_negative

        A_positive = [C[:, 0].min(), C[:, 1].min(), C[:, 2].min()]
        A_negative = [C[:, 0].max(), C[:, 1].max(), C[:, 2].max()]
        S_positive = np.zeros((Nrofrows, 1))
        S_negative = np.zeros((Nrofrows, 1))
        C_value = np.zeros((Nrofrows, 1))

        # Vektor_ID_optimala=np.zeros((1,5))

        for i in range(0, Nrofrows):
            S_positive[i] = (
                (C[i, 0] - A_positive[0]) ** 2
                + (C[i, 1] - A_positive[1]) ** 2
                + (C[i, 2] - A_positive[2]) ** 2
            ) ** 0.5
            S_negative[i] = (
                (C[i, 0] - A_negative[0]) ** 2
                + (C[i, 1] - A_negative[1]) ** 2
                + (C[i, 2] - A_negative[2]) ** 2
            ) ** 0.5
            C_value[i] = S_negative[i] / (S_negative[i] + S_positive[i])

        Best_value = C_value.max()

        # ta fram vilken rad i C_vektorn som har det största värdet
        Row_best_option = np.argmax(C_value)

        # ta fram vilket ingående ID lösningen har
        Vektor_ID_optimala = np.array(ID_vektor[Row_best_option]).tolist()
        IDpoints.append(int(max(Vektor_ID_optimala)))

    return IDpoints


def generate_pareto_cases(problem_constants):
    crm = get_case_reader(problem_constants)
    input_points = list(generate_valid_points(problem_constants, crm))
    pareto_points, dominated_points, dp, pp = calculate(input_points, dominates)
    my_weights = np.matrix(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )  # Weights used to pick points from the ParetoFront
    pareto_case_ids = WeightPPpoints(pp, my_weights)

    for i in pareto_case_ids:
        yield crm.get_case(i)
