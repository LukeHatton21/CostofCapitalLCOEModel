import numpy as np
import warnings
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import pandas as pd
import os 
import regionmask
import warnings
warnings.filterwarnings("ignore")

def get_supply_curves_v3(postprocessor, data, technology, offshore=None, LCOE_cutoff=None):

    # Extract required parameters
    annual_production = data['electricity_production']
    latitudes = annual_production.latitude.values
    longitudes = annual_production.longitude.values

    # Get area and utilisations
    grid_areas = postprocessor.get_areas(annual_production)
    utilisation_factors = postprocessor.get_utilisations(annual_production, technology)

    # Set out constants
    if technology == "Onshore Wind":
        power_density = 6520 # kW/km2
        cutoff = 0.18
    elif technology == "Offshore Wind":
        power_density = 4000 # kW/km2
        cutoff = 0.18
    elif technology == "Solar":
        power_density = 32950  # kW/km2
        cutoff = 0.1
    installed_capacity = 1000

    # Apply cut off factors
    utilisation_factors = xr.where(data['CF']<cutoff, 0, utilisation_factors)
    if LCOE_cutoff is not None:
        data['Calculated_LCOE'] = xr.where(data['CF']<cutoff, np.nan, data['Calculated_LCOE'])
        data['Uniform_LCOE'] = xr.where(data['CF']<cutoff, np.nan, data['Uniform_LCOE'])

    # Scale annual electricity production by power density
    max_installed_capacity = power_density * grid_areas['area'] * utilisation_factors
    ratios = max_installed_capacity / installed_capacity
    technical_potential = annual_production * ratios

    # Include additional data into the dataset
    data['technical_potential'] = technical_potential
    data['capacity_GW'] = max_installed_capacity / 1e+06 # convert from kW to GW 
    if technology == "Offshore Wind":
        data['Country'] = postprocessor.country_grids['sea']
    else:
        data['Country'] = postprocessor.country_grids['land']

    return data

def get_total_generation(wacc_model):
        
    # Extract generation data
    generation_data = wacc_model.CoCModel_class.generation_data

    # Extract specific year
    generation_subset = generation_data[(generation_data['Year'] == 2023) & (generation_data['Variable'] == "Total Generation") & (generation_data['Unit'] == "TWh")] 
    generation_subset_2022 = generation_data[(generation_data['Year'] == 2022) & (generation_data['Variable'] == "Total Generation") & (generation_data['Unit'] == "TWh")] 
    extracted_2023 = generation_subset[["Area", "Country code", "Year", "Continent", "Value"]]
    extracted_2022 = generation_subset[["Area", "Country code", "Year", "Continent", "Value"]]
    extracted_data = extracted_2023
    
    # Extract specific columns that are required
    data_for_output = extracted_data[["Area", "Country code", "Year", "Continent", "Value"]]
    data_for_output = pd.merge(wacc_model.postprocessor.country_mapping.rename(columns={"index":"Country"})[['Country', 'Country code']], extracted_data, how="left", on="Country code")
    data_for_output = data_for_output.rename(columns={"Value":"total_generation"})
    data_for_output.to_csv("TestOutput.csv")

    return data_for_output

def get_abatement_potential(wacc_model, tech_potential_ds,technology):
    
    # Convert to dataframe
    tech_potential_df = tech_potential_ds.to_dataframe()
    
    # Add in region and country to the dataframe
    potential_df_unindexed = pd.merge(tech_potential_df.reset_index(), wacc_model.postprocessor.country_mapping.rename(columns={"index":"Country"})[['Country', 'Region']], how="left", on="Country")
    potential_df = potential_df_unindexed.set_index(["latitude", "longitude"])
    potential_df_reindexed  = potential_df[~potential_df.index.duplicated()]
    
    # Get deployment limits
    country_deployment_limits = get_total_generation(wacc_model)
    
    # Process deployment limits based on solar and wind targets
    country_deployment_limits['total_generation'] = country_deployment_limits['total_generation'] * 1e+09
    country_deployment_limits['total_generation'] = country_deployment_limits['total_generation'].fillna(0)
    
    # Calculate with national limits
    processed_df = apply_national_bounds(potential_df_reindexed.reset_index(), country_deployment_limits)
    
    # Set indexes
    processed_df = processed_df.set_index(["latitude", "longitude"])
    
    # Remove duplicated indexes 
    processed_df = processed_df[~processed_df.index.duplicated()]
    
    # Convert to xarray
    processed_ds = processed_df.to_xarray()
    
    return processed_ds
 
def apply_national_bounds(potential_df, country_deployment_limits):
    
    # Order by country level LCOE
    ratio = 0.25
    potential_df = potential_df.sort_values(by=["Country", "Calculated_LCOE"])
    
    # Extract cumulative capacity potential 
    potential_df["technical_potential_adj"] = potential_df["technical_potential"] * ratio
    potential_df['national_cumulative_potential'] = potential_df.groupby('Country')['technical_potential_adj'].cumsum()
    storage_df = pd.merge(potential_df, country_deployment_limits, how="left", on="Country")

        
    # Create a copy
    processing_df = storage_df.copy()
    
    # Apply limits to each country
    processing_df['bounded_technical_potential'] = np.where(processing_df['national_cumulative_potential'] > processing_df['total_generation'], np.nan, processing_df['technical_potential_adj'])
    processing_df['bounded_abatement'] = np.where(processing_df['national_cumulative_potential'] > processing_df['total_generation'], np.nan, processing_df['abatement']*ratio)
    
    return processing_df
                                 

def calculate_abatement_costs(CI_data, capacity_data):
    
    # Perform the calculation (converting from gCO2/kWh -> tCO2 / kWh)
    tCO2 = CI_data /1000000 * capacity_data['technical_potential'] * 20 * 0.5
    tCO2_MW = CI_data /1000000 * capacity_data['electricity_production'] * 20 * 0.5
    capacity_data["abatement"] = tCO2
    capacity_data["abatement_MW"] = tCO2_MW
    
    return capacity_data

def produce_lcoe_abatement(supply_ds, filename=None, graphmarking=None, position=None, uniform_value=None, technology=None, region_code=None, subnational=None, xlim=None, postprocessor=None):

        def thousands_format(x, pos):
            return f'{int(x):,}'
        
        # Find the index and corresponding cumulative potential for country WACC
        def find_cumulative_at_lcoe(lcoe_series, potential_series, target):
            """
            Finds the cumulative potential at the point where LCOE reaches the target value.

            """
            # Ensure the series are sorted by LCOE
            sorted_indices = lcoe_series.sort_values().index
            lcoe_sorted = lcoe_series[sorted_indices]
            potential_sorted = potential_series[sorted_indices]

            # Interpolate to find the cumulative potential at the target LCOE
            cumulative_at_target = np.interp(target, lcoe_sorted, potential_sorted)
            return cumulative_at_target
        
        
        def get_cumulative_potential(region, target_lcoe, wacc_grouped, uniform_grouped):
            country_cumulative_at_target = find_cumulative_at_lcoe(
                wacc_grouped['Calculated_LCOE'].get_group(region),
                wacc_grouped['cumulative_potential'].get_group(region),
                target_lcoe
            )

            # Get the cumulative potential at the target LCOE for uniform WACC
            uniform_cumulative_at_target = find_cumulative_at_lcoe(
                uniform_grouped['Uniform_LCOE'].get_group(region),
                uniform_grouped['cumulative_potential'].get_group(region),
                target_lcoe
            )

            # Get maximums
            uniform_cumulative_max = np.nanmax(uniform_grouped['cumulative_potential'].get_group(region))
            country_cumulative_max = np.nanmax(wacc_grouped['cumulative_potential'].get_group(region))

        # Convert the dataset into a dataframe
        merged_supply_df = supply_ds.to_dataframe()
        
        # Selecy locations only with bounded abatement
        merged_supply_df = merged_supply_df.loc[~merged_supply_df["bounded_abatement"].isnull()]
        merged_supply_df.to_csv("Test.csv")

        # Merge with country_mapping to give
        supply_df = merged_supply_df.dropna(axis=0,subset=["Calculated_LCOE", "Country"], how="all")

        # Create two copies
        uniform_df = merged_supply_df.dropna(axis=0, subset=["Uniform_LCOE", "Country"], how="all")
        wacc_df = merged_supply_df.dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")
        if subnational is not None:
            subnational_df = merged_supply_df.dropna(axis=0, subset=["Subnational_LCOE", "Country"], how="all")

        # Convert units to TWh
        supply_df['bounded_abatement'] = supply_df['bounded_abatement'] / 1e+09
        uniform_df['bounded_abatement'] = uniform_df['bounded_abatement'] / 1e+09
        wacc_df['bounded_abatement'] = wacc_df['bounded_abatement'] / 1e+09
        if subnational is not None:
            subnational_df['bounded_abatement'] = subnational_df['bounded_abatement'] / 1e+09

        # For the WACC case, sort values and calculate cumulative sum
        wacc_sorted_df = wacc_df.sort_values(by=['Region', 'Calculated_LCOE'], ascending=True)
        wacc_sorted_df['cumulative_potential'] = wacc_sorted_df.groupby('Region')['bounded_abatement'].cumsum()
        wacc_grouped = wacc_sorted_df.groupby('Region')

        # For the Uniform WACC case, sort values and calculate cumulative sum
        uniform_sorted_df = uniform_df.sort_values(by=['Region', 'Uniform_LCOE'], ascending=True)
        uniform_sorted_df['cumulative_potential'] = uniform_sorted_df.groupby('Region')['bounded_abatement'].cumsum()
        uniform_grouped = uniform_sorted_df.groupby('Region')

        if subnational is not None:
            # For the Uniform WACC case, sort values and calculate cumulative sum
            subnational_sorted_df = subnational_df.sort_values(by=['Region', 'Uniform_LCOE'], ascending=True)
            subnational_sorted_df['cumulative_potential'] = subnational_sorted_df.groupby('Region')['bounded_abatement'].cumsum()
            subnational_grouped = subnational_sorted_df.groupby('Region')

        # Set regional colour scheme
        region_colors = {
        "AFR": "purple",          # Stays the same, distinct
        "AUS": "dodgerblue",      # Changed from "cornflowerblue" to a more vivid blue
        "FSU": "gold",            # Changed from "yellow" to "gold" for richer contrast
        "CAN": "lightgray",       # Changed from "silver" to "lightgray" for a softer tone
        "CHN": "red",             # Stays the same, highly distinct
        "CSA": "forestgreen",     # Changed from "green" to "forestgreen" for a deeper tone
        "IND": "darkorange",      # Changed from "orange" to "darkorange" for higher contrast
        "JPN": "teal",            # Changed from "cyan" to "teal" for a stronger, less bright tone
        "MEA": "goldenrod",       # Changed from "olive" to "goldenrod" for a less muted yellow
        "MEX": "black",           # Stays the same, highly distinct
        "ODA": "hotpink",         # Changed from "pink" to "hotpink" for vibrancy
        "EEU": "limegreen",       # Changed from "darkgreen" to "limegreen" for brightness
        "KOR": "chocolate",       # Changed from "sandybrown" to "chocolate" for richer tone
        "USA": "firebrick",       # Changed from "crimson" to "firebrick" for a slightly muted red
        "WEU": "navy"             # Changed from "darkblue" to "navy" for stronger differentiation
        }
        
        if technology == "Solar":
            target_lcoe = 60
        elif technology == "Offshore Wind":
            target_lcoe = 106
        else:
            target_lcoe = 75
                               
        if technology == "Solar":
            min_lcoe = 24
            average_lcoe = 60 
            max_lcoe = 96
        elif technology == "Offshore Wind":
            min_lcoe = 72
            average_lcoe = 106
            max_lcoe = 140
        else:
            min_lcoe = 24
            average_lcoe = 49 
            max_lcoe = 75
        
        # Plot the results
        if technology == "Offshore Wind":
            width_ratio = [1, 1]
        else:
            width_ratio = [1, 3]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8),gridspec_kw = {'wspace':0, 'hspace':0, 'width_ratios': width_ratio})
        region_list = ["AFR", "FSU", "CSA", "MEA", "ODA", "EEU"]
        other_regions = list(set(region_colors.keys()) - set(region_list))
        selected_region_colors = dict((region, region_colors[region]) for region in region_list if region in region_colors)
        other_region_colors = dict((region, region_colors[region]) for region in other_regions if region in region_colors)

        # Loop over the regions specified
        for region in region_list:

            # Extract color for region
            region_shading = region_colors.get(region, "grey")
            
            # Print data
            get_cumulative_potential(region, target_lcoe, wacc_grouped, uniform_grouped)


            # Get cumulative production data for the region
            country_wacc_potential = wacc_grouped[['cumulative_potential']].get_group(region)
            uniform_wacc_potential = uniform_grouped[['cumulative_potential']].get_group(region)

            # Get corresponding lcoe
            country_wacc_lcoe = wacc_grouped[['Calculated_LCOE']].get_group(region)
            uniform_wacc_lcoe = uniform_grouped[['Uniform_LCOE']].get_group(region)
            
            # Conduct interpolation
            country_wacc_potential = country_wacc_potential.interpolate(method="linear")
            country_wacc_lcoe = country_wacc_lcoe.interpolate(method="linear")
            uniform_wacc_potential = uniform_wacc_potential.interpolate(method="linear")
            uniform_wacc_lcoe = uniform_wacc_lcoe.interpolate(method="linear")

            # Produce plots of LCOE against supply for each region
            ax1.plot(country_wacc_potential.iloc[0:-10], country_wacc_lcoe.iloc[0:-10], color=region_shading, linestyle="-")
            ax1.plot(uniform_wacc_potential.iloc[0:-10], uniform_wacc_lcoe.iloc[0:-10], color=region_shading, linestyle="--")

            # Plot the subnational if applicable
            if subnational is not None:
                subnational_wacc_potential = subnational_grouped[['cumulative_potential']].get_group(region)
                subnational_wacc_lcoe = subnational_grouped[['Subnational_LCOE']].get_group(region)
                ax1.plot(subnational_wacc_potential,subnational_wacc_lcoe, color=region_shading, linestyle=":")

        # Create the legend
        handles = [plt.Line2D([0], [0], color=color, lw=4, label=region) for region, color in region_colors.items()]
        ax2.legend(handles=handles, title="Regions", loc=position+ " left", ncol=5, fontsize=12, title_fontsize=15)

        # Set labels
        ax1.set_ylim(0, 250)
        ax2.set_xlim(0, 8)
        ax1.set_ylabel('Levelised Cost of Electricity\n(USD/MWh, '+ technology + ')', fontsize=18)
        ax1.set_xlabel('Developing Regions\nAbatement Potential (GtCO2)', fontsize=18)
        ax1.xaxis.set_major_formatter(FuncFormatter(thousands_format))

        # Set the size of x and y-axis tick labels
        ax1.tick_params(axis='x', labelsize=15)  # Adjust the labelsize as needed
        ax1.tick_params(axis='y', labelsize=15)  # Adjust the labelsize as needed

        # Loop over the regions specified
        for region in other_regions:

            # Extract color for region
            region_shading = region_colors.get(region, "grey")
            
            # Print data
            get_cumulative_potential(region, target_lcoe, wacc_grouped, uniform_grouped)

            # Get cumulative production data for the region
            country_wacc_potential = wacc_grouped[['cumulative_potential']].get_group(region)
            uniform_wacc_potential = uniform_grouped[['cumulative_potential']].get_group(region)

            # Get corresponding lcoe
            country_wacc_lcoe = wacc_grouped[['Calculated_LCOE']].get_group(region)
            uniform_wacc_lcoe = uniform_grouped[['Uniform_LCOE']].get_group(region)

            # Produce plots of LCOE against supply for each region
            ax2.plot(country_wacc_potential,country_wacc_lcoe , color=region_shading, linestyle="-")
            ax2.plot(uniform_wacc_potential,uniform_wacc_lcoe, color=region_shading, linestyle="--")

            # Plot the subnational if applicable
            if subnational is not None:
                subnational_wacc_potential = subnational_grouped[['cumulative_potential']].get_group(region)
                subnational_wacc_lcoe = subnational_grouped[['Subnational_LCOE']].get_group(region)
                ax2.plot(subnational_wacc_potential,subnational_wacc_lcoe, color=region_shading, linestyle=":")

        # Create the legend
        solid_line = plt.Line2D([0], [0], color="black", lw=4, linestyle='-', label='Estimated country- and\ntechnology- WACCs')
        dashed_line = plt.Line2D([0], [0], color="black", lw=4, linestyle='--', label=f'Uniform {uniform_value:0.1f}% WACC')
        style_handles = [solid_line, dashed_line]
        ax1.legend(handles=style_handles, title="Cost of Capital", loc=position+ " right", ncol=1, fontsize=12, title_fontsize=15)
        ax1.spines['left'].set_position('zero')
            
        # Add shading
        ax1.axhspan(min_lcoe, average_lcoe, color="lightblue", alpha=0.1)
        ax2.axhspan(min_lcoe, average_lcoe, color="lightblue", alpha=0.1)
        ax1.text(0.85*6, (min_lcoe+average_lcoe)/2, "US lower\nLCOE range", fontsize=10, ha="center", va="center")

        # Set labels
        ax2.set_ylim(0, 250)
        ax2.set_xlim(0, xlim)
        ax2.set_xlabel('Developed & Industrialising Regions\nAbatement Potential (GtCO2)', fontsize=18)
        ax2.xaxis.set_major_formatter(FuncFormatter(thousands_format))

        # Set the size of x and y-axis tick labels
        ax2.tick_params(axis='x', labelsize=15)  # Adjust the labelsize as needed
        ax2.tick_params(axis='y', labelsize=0)  # Adjust the labelsize as needed

        if xlim is not None:
            if technology == "Offshore Wind":
                ax1.xaxis.set_ticks(np.arange(0, 6, 1))
                ax2.xaxis.set_ticks(np.arange(0, (xlim+5), 5))
            else:
                ax1.xaxis.set_ticks(np.arange(0, 6, 1))
                ax2.xaxis.set_ticks(np.arange(0, (xlim+5), 5))

        if graphmarking is not None:
            ax1.text(0.08, 0.94, graphmarking, transform=ax1.transAxes, fontsize=20, fontweight='bold')

        if region_code is not None:
            ax1.text(0.15, 0.9, technology + "\n" + region_code, transform=ax1.transAxes, fontsize=20, fontweight='bold', ha="center", va="center")

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        plt.show()

def calculate_abatement_potential(results, wacc_model, technology, xlim, filename):
    
    # 1. Calculate abatement costs at each location
    ember_data = pd.read_csv("./DATA/Ember Yearly Data 2024.csv")
    country_CI = extract_carbon_intensity(wacc_model, ember_data)
    country_CI = country_CI["Electricity_CI"]
    results_with_abatement = calculate_abatement_costs(country_CI, results)
    
    # 1b. Plot abatement at each location
    wacc_model.postprocessor.plot_data_shading(results_with_abatement["abatement_MW"]/1000, results_with_abatement.latitude, results_with_abatement.longitude, tick_values=[0, 5, 10, 15, 20], graphmarking="", cmap="Purples", title=technology + "\nLifetime\nAbatement\nPotential\n(kCO2)\n", filename=technology+"_lifetime_abatement")   
    
    # 2. Get supply curves
    supply_df = get_supply_curves_v3(wacc_model.postprocessor, data=results_with_abatement, technology=technology)
    
    # 3. Get abatement potential
    processed_ds = get_abatement_potential(wacc_model, supply_df, technology)
    
    # 4. Plot abatement potential
    if technology == "Solar":
        uniform_value = wacc_model.solar_uf_nom
        graphmarking = "a"
    else:
        uniform_value = wacc_model.onshore_uf_nom
        graphmarking = "b"
    produce_lcoe_abatement(processed_ds, graphmarking=graphmarking, uniform_value=uniform_value, technology=technology, position="upper", xlim=xlim, filename=filename)
    
    return results_with_abatement
    
def extract_carbon_intensity(wacc_model, ember_data):
    
    # Extract carbon intensity data
    ci_data = ember_data.loc[(ember_data["Unit"] == "gCO2/kWh") & (ember_data["Year"] == 2023)]

    # Get columns required
    ci_data_subset = ci_data[["Country code", "Value"]]
    
    # Merge onto country mapping
    ci_data_mapping = pd.merge(wacc_model.country_mapping, ci_data_subset, how="left", on="Country code")
    
    # Merge onto country grids
    country_grids = wacc_model.country_grids.rename({"land":"index"})
    country_df = country_grids.to_dataframe().reset_index()
    country_df = country_df.merge(ci_data_mapping.rename(columns={"Value":"Electricity_CI"}), how="left", on="index")
    country_ds = country_df.drop_duplicates(subset=["latitude", "longitude"]).set_index(["latitude", "longitude"]).to_xarray()
    country_ds = country_ds.assign_coords({"latitude":country_ds.latitude, "longitude": country_ds.longitude})
    
    # Produce heatmap 
    wacc_model.postprocessor.plot_data_shading(country_ds["Electricity_CI"], country_ds.latitude, country_ds.longitude, tick_values=[0, 250, 500, 750, 1000], graphmarking="", cmap="Purples", title="Grid Carbon\nIntensity\n(gCO2/kWh,\n2023)\n", filename="Grid_Carbon_Intensity_2023")   
    
    return country_ds
   
    

abatement_solar = calculate_abatement_potential(wacc_model.solar_lcoe, wacc_model, "Solar", xlim=60, filename="Solar_Abatement_SUBMISSION")
abatement_wind = calculate_abatement_potential(wacc_model.wind_lcoe, wacc_model, "Onshore Wind", xlim=60, filename="Wind_Abatement_SUBMISSION")