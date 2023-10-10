import requests
from bs4 import BeautifulSoup
import pandas as pd

# Initialize an empty list to store the data
all_data = []

# Loop through years and rounds
for year in range(2010, 2024):  # Replace with the range of years you're interested in
    for round_num in range(1, 25):  # Go up to 24 rounds, adjust as needed
        # Construct the URL
        url = f'https://www.footywire.com/afl/footy/supercoach_round?year={year}&round={round_num}&p=&s=T'
        
        # Send an HTTP request to the URL
        response = requests.get(url)
        
        # Check if the page exists (HTTP status code 200 means OK)
        if response.status_code != 200:
            print(f"Skipping year {year}, round {round_num} due to HTTP error.")
            continue
        
        # Parse the HTML content of the page with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table containing the scores
        table = soup.find('table', {'id': 'supercoach-content-table'})
        
        # Check if the table exists on the page
        if table is None:
            print(f"Skipping year {year}, round {round_num} due to missing data.")
            continue
        
        # Loop through each row in the table, skipping the header row
        for row in table.find_all('tr')[1:]:
            # Extract the unique identifier from the 'id' attribute of the <tr> tag
            unique_id = row.get('id', 'N/A')
            
            # Find all columns in the row
            cols = row.find_all('td')
            
            # Extract text from each column
            cols = [col.text.strip() for col in cols]
            
            # Add year, round information, and unique identifier
            cols.extend([year, round_num, unique_id])
            
            # Append to the list
            all_data.append(cols)

# Convert the list to a Pandas DataFrame for further analysis
all_players_by_round_df = pd.DataFrame(all_data, columns=['Rank', 'Player', 'Team', 'CurrentSalary', 'RoundSalary', 'RoundScore', 'RoundValue', 'Year', 'Round', 'UniqueID'])

# Drop unnecessary columns
all_players_by_round_df.drop(['Rank', 'CurrentSalary', 'RoundSalary', 'RoundValue'], axis=1, inplace=True)

# Drop rows where 'UniqueID' is blank
all_players_by_round_df.dropna(subset=['UniqueID'], inplace=True)

# Keep only the numeric values after the underscore in the 'UniqueID' column
all_players_by_round_df['UniqueID'] = all_players_by_round_df['UniqueID'].str.extract('_(\d+)')

# Convert 'UniqueID' to a whole number
all_players_by_round_df['UniqueID'] = all_players_by_round_df['UniqueID'].astype(int)

# Create a master DataFrame with unique player IDs and names
player_master_df = all_players_by_round_df[['UniqueID', 'Player']].drop_duplicates()

# Convert 'UniqueID' in player_master_df to a whole number
player_master_df['UniqueID'] = player_master_df['UniqueID'].astype(int)

# Save the DataFrames to CSV files
all_players_by_round_df.to_csv('All_Players_By_Round_Data.csv', index=False)
player_master_df.to_csv('Player_Master_Data.csv', index=False)

# Initialize an empty list to store player names and their corresponding pids
player_names_data = []

# Loop through pid values from 100 to 300
for pid in range(100, 5000):
    # Construct the URL for each player's profile
    profile_url = f'https://www.footywire.com/afl/footy/ft_player_profile?pid={pid}'
    
    # Send an HTTP request to the URL
    response = requests.get(profile_url)
    
    # Check if the page exists (HTTP status code 200 means OK)
    if response.status_code != 200:
        print(f"Skipping pid {pid} due to HTTP error.")
        continue
    
    # Parse the HTML content of the page with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the player's name using the 'id' attribute
    player_name_tag = soup.find('h3', {'id': 'playerProfileName'})
    
    # Check if the player name exists on the page
    if player_name_tag is None:
        print(f"Skipping pid {pid} due to missing data.")
        continue
    
    # Extract the player's name
    player_name = player_name_tag.text.strip()
    
    # Append to the list
    player_names_data.append([pid, player_name])

# Convert the list to a Pandas DataFrame
player_name_df = pd.DataFrame(player_names_data, columns=['PID', 'PlayerName'])

# Save the DataFrame to a CSV file
player_name_df.to_csv('Player_Name_Data.csv', index=False)


# Perform your analysis here

