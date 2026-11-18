import csv

with open("datafiles/Basler_a2A2464_18nov-meting1.mqa", "r", newline="") as f, \
     open("datafiles/Basler_a2A2464_18nov-meting1_CSV.mqa", "w", newline="") as g:

    tab_file = csv.reader(f, dialect=csv.excel_tab)
    comma_file = csv.writer(g, dialect=csv.excel)

    for row in tab_file:
        comma_file.writerow(row)