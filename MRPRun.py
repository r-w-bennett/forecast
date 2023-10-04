import pandas as pd
import math
import matplotlib.pyplot as plt

# Sample DataFrames (To Be Replaced with Actuals)
item_master_data = pd.DataFrame({
    'Item': ['Breast 500g Tray', 'Drumstick 500g Tray', 'Breast', 'Drumstick', 'Tray'],
    'Lead Time': [0, 0, 1, 1, 1],
    'Order Type': [0, 0, 1, 1, 1],
    'MoQ': [5, 10, 5, 20, 10],
    'Safety Stock': [20, 10, 20, 10, 20],
    'Putaway Time': [0, 0, 1, 1, 1]
})

bom_data = pd.DataFrame({
    'Parent Item': ['Breast 500g Tray', 'Breast 500g Tray', 'Drumstick 500g Tray', 'Drumstick 500g Tray'],
    'Child Item': ['Breast', 'Tray', 'Drumstick', 'Tray'],
    'Ratio': [2, 1, 2, 1]
})

inventory_data = pd.DataFrame({
    'Item': ['Breast 500g Tray', 'Drumstick 500g Tray', 'Breast', 'Drumstick', 'Tray'],
    'Inventory': [25, 30, 40, 35, 50]
})

weeks = list(range(0, 13))
demand_over_time_data = pd.DataFrame([{'Item': 'Breast 500g Tray', 'Demand': 20 + week, 'Week': week} for week in weeks] +
                                    [{'Item': 'Drumstick 500g Tray', 'Demand': 15 + week, 'Week': week} for week in weeks])

existing_orders_data = pd.DataFrame([{'Item': item, 'Order': 40 if item == 'Breast' else 35 if item == 'Drumstick' else 45, 'Week': week} 
                                     for week in range(1, 5) for item in ['Breast', 'Drumstick', 'Tray']])

# Generate a DataFrame containing all combinations of items and weeks
all_items = item_master_data['Item'].unique()
all_weeks = list(range(0, 13))
all_combinations = pd.DataFrame([(item, week) for item in all_items for week in all_weeks],
                                columns=['Item', 'Week'])

# Create pivot table from demand_over_time_data and then reset the index
pivot = pd.pivot_table(demand_over_time_data, values='Demand', index=['Item'], 
                       columns=['Week'], fill_value=0).reset_index()

# Melt the pivot table to go back to long format
melted_pivot = pivot.melt(id_vars=['Item'], value_vars=all_weeks, 
                          var_name='Week', value_name='Demand')

# Merge the all_combinations dataframe with the melted pivot
initial_demand_profile = pd.merge(all_combinations, melted_pivot, on=['Item', 'Week'], how='left')

# Fill NaN values in the 'Demand' column with 0
initial_demand_profile['Demand'] = initial_demand_profile['Demand'].fillna(0).astype(int)

# Left outer join with item_master_data to add the required columns
initial_demand_profile = pd.merge(initial_demand_profile, item_master_data[['Item', 'Lead Time', 'Order Type', 'MoQ', 'Safety Stock', 'Putaway Time']], 
                                  on='Item', how='left')

# Sort by 'Item' (ascending) and 'Week' (ascending)
initial_demand_profile = initial_demand_profile.sort_values(by=['Item', 'Week']).reset_index(drop=True)

# Left outer join with existing_orders_data to add the 'Order' column
initial_demand_profile = pd.merge(initial_demand_profile, existing_orders_data[['Item', 'Week', 'Order']], 
                                  on=['Item', 'Week'], how='left')

# Fill NaN values in the 'Order' column with 0
initial_demand_profile['Order'] = initial_demand_profile['Order'].fillna(0).astype(int)

# Calculation Function for Planned Orders
def order_and_available_calc(df, inventory_data):
    # Initial empty lists to store results
    sohs = []
    planned_orders = []
    availables = []
    
    # Lists to store data for 'Planned Order Receipt' and 'Planned Order Release'
    planned_order_receipt_list = []
    planned_order_release_list = []

    # Group the input dataframe by 'Item' and process each group
    for item, group in df.groupby('Item'):
        # Get the current inventory level for the item
        current_inventory = inventory_data[inventory_data['Item'] == item]['Inventory'].values[0]
        
        # Iterate over rows in the group
        for i, row in group.iterrows():
            # For the earliest available week, set SOH to current inventory
            if i == group.index[0]:
                soh = current_inventory
            else:
                soh = availables[-1]  # Set to 'Available To Promise' from previous week

            demand = row['Demand']
            order = row['Order']
            safety_stock = row['Safety Stock']
            moq = row['MoQ']

            # Calculate tentative availability
            tentative_avail = soh - demand + order

            # Calculate Planned Order
            if (row['Week'] - (row['Putaway Time'] + row['Lead Time']) >= 0):
                if tentative_avail < safety_stock:
                    deficit = safety_stock - tentative_avail
                    planned_order = math.ceil(deficit / moq) * moq
                else:
                    planned_order = 0
            else:
                planned_order = 0

            # Calculate Available To Promise
            available = soh - demand + order + planned_order
            
            # Append results to lists
            sohs.append(soh)
            planned_orders.append(planned_order)
            availables.append(available)

            # Calculate 'Planned Order Receipt' and 'Planned Order Release'
            planned_order_receipt_list.append(
                (item, row['Week'] - row['Putaway Time'], planned_order)
            )
            planned_order_release_list.append(
                (item, row['Week'] - (row['Putaway Time'] + row['Lead Time']), planned_order)
            )

    # Convert the temporary lists to dataframes
    df_receipt = pd.DataFrame(planned_order_receipt_list, columns=['Item', 'Week', 'Planned Order Receipt'])
    df_release = pd.DataFrame(planned_order_release_list, columns=['Item', 'Week', 'Planned Order Release'])
    
    # Remove rows where 'Week' is less than zero
    df_receipt = df_receipt[df_receipt['Week'] >= 0]
    df_release = df_release[df_release['Week'] >= 0]

    # Add the lists as new columns to the dataframe
    df['SOH'] = sohs
    df['Available To Promise'] = availables

    # Merge with 'Planned Order Receipt' and 'Planned Order Release' dataframes
    df = pd.merge(df, df_receipt, on=['Item', 'Week'], how='left')
    df = pd.merge(df, df_release, on=['Item', 'Week'], how='left')
    
    # Replace NaN values with 0
    df['Planned Order Receipt'].fillna(0, inplace=True)
    df['Planned Order Release'].fillna(0, inplace=True)
    
    return df

# Call the function
initial_result_df = order_and_available_calc(initial_demand_profile, inventory_data)

# Create final_result_df by filtering on Order Type = 1 and selecting specific columns
final_result_df = initial_result_df[initial_result_df['Order Type'] == 1][['Item', 'Week', 'Demand', 'Lead Time', 'Order Type', 'MoQ', 'Safety Stock', 'Putaway Time', 'Order']]

# Create explosion_df by filtering on Order Type = 0 and selecting specific columns
explosion_df = initial_result_df[initial_result_df['Order Type'] == 0][['Item', 'Week', 'Demand', 'Lead Time', 'Order Type', 'MoQ', 'Safety Stock', 'Putaway Time', 'Order', 'Planned Order Release']]

# Create an empty dataframe to hold processed demand
processed_demand = pd.DataFrame()

while not explosion_df.empty:
    # Step 1: Concatenate rows with order type = 1 to processed_demand
    processed_demand = pd.concat([processed_demand, explosion_df[explosion_df['Order Type'] == 1]])
    
    # Drop these rows from explosion_df
    explosion_df = explosion_df[explosion_df['Order Type'] != 1]

    # Step 2: Check if explosion_df is empty, and if so, exit the loop
    if explosion_df.empty:
        break
    
    # Step 3: Change Order Type values
    explosion_df['Order Type'] = 1

    # Insert rows for child items
    child_rows = []
    for _, row in explosion_df.iterrows():
        parent_item = row['Item']
        week = row['Week']
        parent_order_release = row['Planned Order Release']

        for _, bom_row in bom_data[bom_data['Parent Item'] == parent_item].iterrows():
            child_item = bom_row['Child Item']
            ratio = bom_row['Ratio']
            child_demand = parent_order_release * ratio
            
            item_data = item_master_data[item_master_data['Item'] == child_item].iloc[0]
            child_order = existing_orders_data[
                (existing_orders_data['Item'] == child_item) & 
                (existing_orders_data['Week'] == week)
            ]['Order'].values[0] if not existing_orders_data[
                (existing_orders_data['Item'] == child_item) & 
                (existing_orders_data['Week'] == week)
            ].empty else 0

            child_rows.append({
                'Item': child_item,
                'Week': week,
                'Demand': child_demand,
                'Lead Time': item_data['Lead Time'],
                'Order Type': item_data['Order Type'],
                'MoQ': item_data['MoQ'],
                'Safety Stock': item_data['Safety Stock'],
                'Putaway Time': item_data['Putaway Time'],
                'Order': child_order,
                'Planned Order Release': 0
            })

    child_df = pd.DataFrame(child_rows)
    explosion_df = pd.concat([explosion_df, child_df])

    # Step 4: Remove 'Planned Order Release' column
    explosion_df.drop('Planned Order Release', axis=1, inplace=True)
    
    # After Step 4, Before Step 5: Group by columns and sum up the Demand
    explosion_df = explosion_df.groupby(['Item', 'Week', 'Lead Time', 'Order Type', 'MoQ', 'Safety Stock', 'Putaway Time', 'Order']).agg({'Demand': 'sum'}).reset_index()
    
    # Step 5: Call order_and_available_calc function
    
    # Step 5: Call order_and_available_calc function
    explosion_df = order_and_available_calc(explosion_df, inventory_data)
    
    # Step 6: Select necessary columns
    explosion_df = explosion_df[['Item', 'Week', 'Demand', 'Lead Time', 'Order Type', 'MoQ', 'Safety Stock', 'Putaway Time', 'Order', 'Planned Order Release']]

# End of the loop

# Drop the 'Planned Order Release' column from processed_demand
processed_demand.drop('Planned Order Release', axis=1, inplace=True)

# Concatenate processed_demand onto final_result_df
final_result_df = pd.concat([final_result_df, processed_demand])

# Group by the provided columns and sum up the Demand
final_result_df = final_result_df.groupby(['Item', 'Week', 'Lead Time', 'Order Type', 'MoQ', 'Safety Stock', 'Putaway Time', 'Order']).agg({'Demand': 'sum'}).reset_index()

# Merge to get the correct 'Order Type' from item_master_data
final_result_df = final_result_df.merge(item_master_data[['Item', 'Order Type']], on='Item', how='left')

# Drop the original 'Order Type' column and rename the new one
final_result_df.drop('Order Type_x', axis=1, inplace=True)
final_result_df.rename(columns={'Order Type_y': 'Order Type'}, inplace=True)


# Call order_and_available_calc on final_result_df to generate mrp_df
mrp_df = order_and_available_calc(final_result_df, inventory_data)

# List of columns to format
cols_to_format = ['Order', 'Demand', 'SOH', 'Available To Promise', 'Planned Order Receipt', 'Planned Order Release']

# Format the columns
for col in cols_to_format:
    mrp_df[col] = mrp_df[col].round(0).astype(int)  # First, round to 0 decimal places and convert to integer
    mrp_df[col] = mrp_df[col].apply(lambda x: '{:,}'.format(x))  # Then, format with commas

# Display the updated DataFrame
print(mrp_df)


import matplotlib.pyplot as plt

def plot_item_tables(df):
    # Get unique items
    items = df['Item'].unique()
    
    for item in items:
        # Filter data for the specific item
        item_data = df[df['Item'] == item].sort_values(by='Week')
        
        # Extract the relevant data
        weeks = item_data['Week'].values
        soh = item_data['SOH'].values
        demand = item_data['Demand'].values
        order = item_data['Order'].values
        planned_order_receipt = item_data['Planned Order Receipt'].values
        planned_order_release = item_data['Planned Order Release'].values
        available_to_promise = item_data['Available To Promise'].values
        
        # Set up the table data
        columns = weeks
        rows = ['SOH', 'Demand', 'Order', 'Planned Order Receipt', 'Planned Order Release', 'Available To Promise']
        data = [soh, demand, order, planned_order_receipt, planned_order_release, available_to_promise]
        
        # Create the table
        fig, ax = plt.subplots(figsize=(10, 4))  # set the size that you'd like (width, height)
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=data, rowLabels=rows, colLabels=columns, cellLoc = 'center', loc='center')
        
        ax.set_title(f'MRP Data for {item}')
        plt.show()

# Call the function
plot_item_tables(mrp_df)


