#Make directory
DDE_BACKEND=pytorch python main.py make_dir
# Read the path_directory from the temporary file
path_directory=$(cat temp_path_directory.txt)
rm temp_path_directory.txt  #remove the temporary file

# Run the Python script with nohup and save the nohup.out in the path_directory
DDE_BACKEND=pytorch nohup python main.py train > "${path_directory}/nohup.out" 2>&1 &
#DDE_BACKEND=pytorch nohup python main.py train_adaptive > "${path_directory}/nohup.out" 2>&1 &
