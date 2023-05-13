url_2023 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/e6013a93-1321-4f2a-bf91-8d8a02f1e62f/download/tmp9g_820k8.csv"
url_2022 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/81a7b022-f8fc-4da5-80e4-b160058ca207/download/tmph4izx_fb.csv"
url_2021 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/f53ebccd-bc61-49f9-83db-625f209c95f5/download/tmppgq9965_.csv"
url_2020 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/6ff6a6fd-3141-4440-a880-6f60a37fe789/download/script_105774672_20210108153400_combine.csv"
url_2019 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/ea2e4696-4a2d-429c-9807-d02eb92e0222/download/311_service_requests_2019.csv"
url_2018 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/2be28d90-3a90-4af1-a3f6-f28c1e25880a/download/311_service_requests_2018.csv"
url_2017 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/30022137-709d-465e-baae-ca155b51927d/download/311_service_requests_2017.csv"
url_2016 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/b7ea6b1b-3ca4-4c5b-9713-6dc1db52379a/download/311_service_requests_2016.csv"
url_2015 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/c9509ab4-6f6d-4b97-979a-0cf2a10c922b/download/311_service_requests_2015.csv"
url_2014 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/bdae89c8-d4ce-40e9-a6e1-a5203953a2e0/download/311_service_requests_2014.csv"
url_2013 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/407c5cd0-f764-4a41-adf8-054ff535049e/download/311_service_requests_2013.csv"
url_2012 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/382e10d9-1864-40ba-bef6-4eea3c75463c/download/311_service_requests_2012.csv"
url_2011 = "https://data.boston.gov/dataset/8048697b-ad64-4bfc-b090-ee00169f2323/resource/94b499d9-712a-4d2a-b790-7ceec5c9c4b1/download/311_service_requests_2011.csv"


# Get a list of all CSV files in the directory
all_files = [url_2023, url_2022, url_2021, url_2020, url_2019, url_2018, url_2017, url_2016, url_2015, url_2014, url_2013, url_2012, url_2011]

# Create an empty list to store the dataframes
dfs = []

# Loop through the files and load them into dataframes
for file in all_files:
  df = pd.read_csv(file)
  dfs.append(df)

#check that the files all have the same number of columns, and the same names
same_list_num_col = []
diff_list_num_col = []
same_list_order_col = []
diff_list_order_col = []

for i in range(len(dfs)):

  if dfs[i].shape[1] != dfs[0].shape[1]:
    #print('Error: File', i, 'does not have the same number of columns as File 0')
    diff_list_num_col.append(i)
  else:
    #print('File', i, 'has same number of columns as File 0')
    same_list_num_col.append(i)
  if not dfs[i].columns.equals(dfs[0].columns):
    #print('Error: File', i, 'does not have the same column names and order as File 0')
    diff_list_order_col.append(i)
  else:
    #print('File', i, 'has the same column name and order as File 0')
    same_list_order_col.append(i)

print("Files with different number of columns from File 0: ", diff_list_num_col)
print("Files with same number of columns as File 0: ", same_list_num_col)
print("Files with different column order from File 0: ", diff_list_order_col)
print("Files with same column order as File 0: ", same_list_order_col)

# Concatenate the dataframes into a single dataframe
df_all = pd.concat(dfs, ignore_index=True)