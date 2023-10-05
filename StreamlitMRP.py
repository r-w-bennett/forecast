import pandas as pd
import math
import matplotlib.pyplot as plt
import streamlit as st
import os
import signal
import altair as alt
from io import StringIO

# Add widgets for user input
bom_data_input = st.text_area("Paste BOM Data (CSV format):")
item_master_data_input = st.text_area("Paste Item Master Data (CSV format):")
inventory_data_input = st.text_area("Paste Inventory Data (CSV format):")
existing_orders_data_input = st.text_area("Paste Existing Orders Data (CSV format):")

# Convert the user input to DataFrames
if bom_data_input:
    bom_data = pd.read_csv(StringIO(bom_data_input))
if item_master_data_input:
    item_master_data = pd.read_csv(StringIO(item_master_data_input))
if inventory_data_input:
    inventory_data = pd.read_csv(StringIO(inventory_data_input))
if existing_orders_data_input:
    existing_orders_data = pd.read_csv(StringIO(existing_orders_data_input))

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
    
# Copy mrp_df to a new DataFrame
mrp_df_copy = mrp_df.copy()

# Convert 'Available To Promise' and 'Planned Order Release' back to integer for comparison
mrp_df_copy['Available To Promise'] = mrp_df_copy['Available To Promise'].str.replace(',', '').astype(int)
mrp_df_copy['Planned Order Release'] = mrp_df_copy['Planned Order Release'].str.replace(',', '').astype(int)

# Filter rows where 'Available To Promise' is less than 0
shortage_df = mrp_df_copy[mrp_df_copy['Available To Promise'] < 0][['Item', 'Week']]

# Create a new column with the shortage message
shortage_df['Message'] = shortage_df.apply(lambda row: f"{row['Item']} shortage in week {row['Week']}", axis=1)

# Filter rows where 'Week' is 0 and 'Planned Order Release' is greater than 0
order_release_df = mrp_df_copy[(mrp_df_copy['Week'] == 0) & (mrp_df_copy['Planned Order Release'] > 0)][['Item', 'Week']]

# Create a new column with the order release message
order_release_df['Message'] = order_release_df.apply(lambda row: f"Order Release Required for {row['Item']} in week {row['Week']}", axis=1)

# Filter rows where 'Available To Promise' is less than 'Safety Stock' and greater than or equal to 0
safety_stock_df = mrp_df_copy[(mrp_df_copy['Available To Promise'] < mrp_df_copy['Safety Stock']) & (mrp_df_copy['Available To Promise'] >= 0)][['Item', 'Week']]

# Create a new column with the safety stock message
safety_stock_df['Message'] = safety_stock_df.apply(lambda row: f"Below Safety Stock for {row['Item']} in week {row['Week']}", axis=1)

# Concatenate the three DataFrames to have all messages
all_messages_df = pd.concat([shortage_df, order_release_df, safety_stock_df], ignore_index=True)

# Now, all_messages_df contains both types of messages, each in its own row.

# Display the updated DataFrame
print(mrp_df)

def plot_item_tables(df, item_master_data, all_messages_df):
    # Get unique items
    items = df['Item'].unique()

    # Create a dropdown menu for item selection
    selected_item = st.selectbox('Select Item', items)

    # Filter data for the selected item
    item_data = df[df['Item'] == selected_item].sort_values(by='Week')

    # Get item master data for the selected item
    item_info = item_master_data[item_master_data['Item'] == selected_item]

    # Create columns
    col1, col2 = st.columns(2)

    # Create tables in the first column
    with col1:
        # Create a table for exceptions (shortages and order releases)
        st.write(f'### Exceptions for {selected_item}')
        exception_data = all_messages_df[all_messages_df['Item'] == selected_item]
        if exception_data.empty:
            st.write("No Exceptions")
        else:
            st.table(exception_data.set_index('Week'))

        # Create a table for item information
        st.write('### Item Information')
        st.table(item_info.set_index('Item'))

        # Transpose MRP data for the selected item to have Week as columns
        relevant_columns = ['Week', 'SOH', 'Demand', 'Order', 'Planned Order Receipt', 'Available To Promise', 'Planned Order Release']
        transposed_item_data = item_data[relevant_columns].set_index('Week').transpose()
        st.write(f'### MRP Data for {selected_item}')
        st.table(transposed_item_data)

    # Convert columns back to integers for plotting
    item_data[["SOH", "Order", "Planned Order Receipt"]] = item_data[["SOH", "Order", "Planned Order Receipt"]].apply(pd.to_numeric, errors='coerce')

    # Create a line and bar chart in the second column
    with col2:
        # Create a line and bar chart
        st.write(f'### MRP Chart for {selected_item}')

        line = alt.Chart(item_data).mark_line(color='blue', size=3).encode(
            x='Week:O',
            y='SOH:Q',
            color=alt.value('blue'),
            tooltip=['Week', 'SOH'],
            opacity=alt.value(1)
        ).properties(
            title='SOH Over Time'
        )

        bar1 = alt.Chart(item_data).mark_bar().encode(
            x='Week:O',
            y='Order:Q',
            color=alt.value('#A8D5BA'),  # Soft green
            tooltip=['Week', 'Order'],
            opacity=alt.value(0.7)
        )

        bar2 = alt.Chart(item_data).mark_bar().encode(
            x='Week:O',
            y='Planned Order Receipt:Q',
            color=alt.value('#7DA87B'),  # Another shade of soft green
            tooltip=['Week', 'Planned Order Receipt'],
            opacity=alt.value(0.7)
        )

        # Layer the charts with the line chart at the front
        chart = alt.layer(bar1, bar2, line).resolve_scale(
            color='independent'
        )

        chart = chart.configure_legend(
            orient='top'
        )

        st.altair_chart(chart, use_container_width=True)

# Streamlit App Configuration
st.set_page_config(layout="wide")

# Streamlit App
st.title('MRP Dashboard')

# Sidebar for navigation
page = st.sidebar.radio(
    'Select a Page',
    ['Exception Messages', 'Item Details']
)

if page == 'Item Details':
    st.title('Item Details')
    plot_item_tables(mrp_df, item_master_data, all_messages_df)
elif page == 'Exception Messages':
    st.title('Exception Messages')
    if all_messages_df.empty:
        st.write("No Exceptions")
    else:
        st.table(all_messages_df.set_index('Week'))

if st.button('Close app'):
    st.write('Closing the app...')
    os.kill(os.getpid(), signal.SIGTERM)