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




class PostProcessor:
    def __init__(self, output_folder, solar_results, wind_results, offshore_results, GDP_capita, land_cover, land_mapping, country_grids, TIAM_regions, country_mapping, solar_cf, wind_cf, country_wacc_mapping):
        """ Initialises the PostProcessor class, which is used for geospatial mapping and postprocessing
       
        Inputs:
        Output_Folder - folder for printing outputs
        Wind_results - Dataset containing LCOE, electricity production and WACC values for Wind
        Solar_results - Dataset containing LCOE, electricity production and WACC values for Solar
        GDP_capita - CSV containing GDP per capita for all countries
        Land_cover - netcdf file with codings for land cover categories
        Land_mapping - CSV file containing land utilisation rates for land cover categories
        TIAM_regions - CSV file containing a mapping of the TIAM regions
        Country_mapping - CSV file with a mapping of country numbers to country codes
        Solar_CF - netcdf file containing average capacity factor
        Wind_CF - netcdf file containing average capacity factor
        country_wacc_mapping - Mapping of country waccs for solar PV, onshore wind and offshore wind
        
        
        """
        
        # Read in the output folder
        self.output_folder = output_folder
        
        # Read in the solar and wind parameters
        self.solar_results = solar_results
        self.wind_results = wind_results
        self.offshore_results = offshore_results
        self.solar_cf = solar_cf
        self.wind_cf = wind_cf
        
        # Collate wind results
        self.collated_wind_results = xr.merge([self.wind_results.drop(labels="capacity_GW", errors="ignore"), self.offshore_results], join="inner", compat="no_conflicts")
        
        # Read in the input parameters
        self.GDP_capita = GDP_capita
        self.land_cover = land_cover
        self.land_mapping = land_mapping
        self.country_grids = country_grids
        self.country_mapping = country_mapping
        self.TIAM_regions = TIAM_regions
        
        # Perform merges for GDP and for country grids
        self.GDP_country_mapping = pd.merge(self.country_mapping, self.GDP_capita, on="Country code", how="left")
        self.country_wacc_mapping = country_wacc_mapping
                 
                 
                 
    def get_utilisations(self, annual_production, technology):

        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values
        global_cover = self.land_cover.reindex_like(annual_production, method="nearest")
        mapping = self.land_mapping

        utilisation = xr.zeros_like(global_cover['cover'])
        for i in np.arange(0, 21, 1):
            # Use xarray's where and isin functions to map land use categories to values
            if technology == "Solar":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
            elif technology =="Onshore Wind":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)
            elif technology == "Offshore Wind":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], 1, utilisation)

        return utilisation    


    def get_areas(self, annual_production):

        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values

        # Add an extra value to latitude and longitude coordinates
        latitudes_extended = np.append(latitudes, latitudes[-1] + np.diff(latitudes)[-1])
        longitudes_extended = np.append(longitudes, longitudes[-1] + np.diff(longitudes)[-1])

        # Calculate the differences between consecutive latitude and longitude points
        dlat_extended = np.diff(latitudes_extended)
        dlon_extended = np.diff(longitudes_extended)

        # Calculate the Earth's radius in kms
        radius = 6371

        # Compute the mean latitude value for each grid cell
        mean_latitudes_extended = (latitudes_extended[:-1] + latitudes_extended[1:]) / 2
        mean_latitudes_2d = mean_latitudes_extended[:, np.newaxis]

        # Convert the latitude differences and longitude differences from degrees to radians
        dlat_rad_extended = np.radians(dlat_extended)
        dlon_rad_extended = np.radians(dlon_extended)

        # Compute the area of each grid cell using the Haversine formula
        areas_extended = np.outer(dlat_rad_extended, dlon_rad_extended) * (radius ** 2) * np.cos(np.radians(mean_latitudes_2d))

        # Create a dataset to store results
        area_dataset = xr.Dataset()
        area_dataset['latitude'] = latitudes
        area_dataset['longitude'] = longitudes
        area_dataset['area'] = (['latitude', 'longitude'], areas_extended, {'latitude': latitudes, 'longitude': longitudes})

        return area_dataset



    def get_supply_curves_v2(self, data, technology, offshore=None, LCOE_cutoff=None):

        # Extract required parameters
        annual_production = data['electricity_production']
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values

        # Get area and utilisations
        grid_areas = self.get_areas(annual_production)
        utilisation_factors = self.get_utilisations(annual_production, technology)

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
        data['capacity_GW'] = max_installed_capacity * utilisation_factors / 1e+06 # convert from kW to GW 
        if technology == "Offshore Wind":
            data['Country'] = self.country_grids['sea']
        else:
            data['Country'] = self.country_grids['land']

        return data

    def produce_wacc_potential_curve_v2(self, supply_ds, filename=None, graphmarking=None, title=None, xlim=None, uniform_value=None, technology=None, region_code=None, subnational=None):

        # Convert the dataset into a dataframe
        supply_df = supply_ds.to_dataframe()

        # Remove locations not evaluated
        supply_df = supply_df.dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")

        # Create two copies
        uniform_df = supply_ds.to_dataframe().dropna(axis=0, subset=["Uniform_LCOE", "Country"], how="all")
        wacc_df = supply_ds.to_dataframe().dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")
        if subnational is not None:
            subnational_df = supply_ds.to_dataframe().dropna(axis=0, subset=["Subnational_LCOE", "Country"], how="all")
        
        # Convert units to TWh
        supply_df['technical_potential'] = supply_df['technical_potential'] / 1e+09
        uniform_df['technical_potential'] = uniform_df['technical_potential'] / 1e+09
        wacc_df['technical_potential'] = wacc_df['technical_potential'] / 1e+09
        if subnational is not None:
            subnational_df['technical_potential'] = subnational_df['technical_potential'] / 1e+09

        # For the Country WACC case, sort values and calculate cumulative sum
        supply_df = supply_df.round({'Estimated_WACC': 3})
        supply_df = supply_df.sort_values(by=['Estimated_WACC'], ascending=True)
        supply_df['cumulative_potential'] = supply_df['technical_potential'].cumsum()

        # For the WACC case, sort values and calculate cumulative sum
        wacc_sorted_df = wacc_df.sort_values(by=['Calculated_LCOE'], ascending=True)
        wacc_sorted_df['cumulative_wacc'] = wacc_sorted_df['technical_potential'].cumsum()

        # For the Uniform WACC case, sort values and calculate cumulative sum
        uniform_sorted_df = uniform_df.sort_values(by=['Uniform_LCOE'], ascending=True)
        uniform_sorted_df['cumulative_uniform'] = uniform_sorted_df['technical_potential'].cumsum()
        uniform_sorted_df = uniform_sorted_df.drop(uniform_sorted_df[uniform_sorted_df['cumulative_uniform'] > np.nanmax( supply_df['cumulative_potential'])].index)
        
        if subnational is not None:
            # For the Uniform WACC case, sort values and calculate cumulative sum
            subnational_sorted_df = subnational_df.sort_values(by=['Subnational_LCOE'], ascending=True)
            subnational_sorted_df['cumulative_uniform'] = subnational_sorted_df['technical_potential'].cumsum()
            subnational_sorted_df = subnational_sorted_df.drop(subnational_sorted_df[subnational_sorted_df['cumulative_uniform'] > np.nanmax( supply_df['cumulative_potential'])].index)
        
        # Print maximums
        print(f"The maximum for supply_df is {np.nanmax(supply_df['cumulative_potential'])}, the maximum for the uniform case is {np.nanmax(wacc_sorted_df['cumulative_wacc'])} and the max for country specific is {np.nanmax(uniform_sorted_df['cumulative_uniform'])}")


        # Plot the results
        fig, ax = plt.subplots(figsize=(20, 8))
        color_labels = {}
        cmap = mpl.colormaps['gnuplot_r']
        norm = mpl.colors.Normalize(vmin=0, vmax=50000)  # Normalize to the range of GDP


        # Iterate through each data point and create a bar with the specified width
        for index, row in supply_df.iterrows():
            width = row['technical_potential']   # Bar width, in TWh
            height = row['Estimated_WACC']  # Bar height
            country = row['Country']
            cumulative_production = row['cumulative_potential'] # Cumulative production, in TWh

            # Get GDP per capita 
            if np.isnan(country):
                gdp_per_capita = np.nan
            else:
                gdp_per_capita = self.GDP_country_mapping.loc[self.GDP_country_mapping['index'] == country, '2024'].values[0]
            if np.isnan(gdp_per_capita) or gdp_per_capita is None or gdp_per_capita == "no data":
                color = "gray"
            else:
                color = cmap(norm(gdp_per_capita))

            # Plot a bar with the specified width, height, x-position, and color
            ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)


        def thousands_format(x, pos):
            return f'{int(x):,}'

        # Set labels
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, 30)
        ax.set_ylabel('WACC (%)', fontsize=20)
        ax.set_xlabel('Annual Electricity Potential (TWh/year)', fontsize=25)
        ax.set_title(title, fontsize=30)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed

        # Add color bar
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 450000,50000], format=',', extend="max", anchor=(0.25, 0.5))
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(thousands_format))
        cbar.set_label('GDP per capita (USD, 2024)', fontsize=20)
        cbar.ax.tick_params(labelsize=15)


        # Plot second axis
        # Create a twin Y-axis on the same figure
        ax_twin = ax.twinx()

        # Plot lines on the twin Y-axis (this axis is independent of the main y-axis)
        ax_twin.plot(wacc_sorted_df['cumulative_wacc'], wacc_sorted_df['Calculated_LCOE'], color='blue', lw=2.5, label='LCOE under Country WACCs', linestyle="--")
        ax_twin.plot(uniform_sorted_df['cumulative_uniform'], uniform_sorted_df['Uniform_LCOE'], color='red', lw=2.5, label=f'LCOE under UNFCCC Annex II Average ({uniform_value:0.1f}%)', linestyle="--")
        if subnational is not None:
            ax_twin.plot(subnational_sorted_df['cumulative_uniform'], subnational_sorted_df['Subnational_LCOE'], color='black', lw=2.5, label=f'LCOE under subnational WACCs', linestyle="--")

        # Customize the second y-axis
        ax_twin.set_ylabel('Levelised Cost (USD/MWh)', fontsize=20)
        ax_twin.legend(loc='upper center', fontsize=20)
        ax_twin.tick_params(axis="y", labelsize=20)
        ax_twin.set_ylim(0, 300)

        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
            
        if region_code is not None:
            ax.text(0.15, 0.9, technology + "\n" + region_code, transform=ax.transAxes, fontsize=20, fontweight='bold', ha="center", va="center")

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        plt.show()

        return supply_df
    
    
    def produce_potential_curve_v3(self, supply_ds, filename=None, graphmarking=None, title=None, uniform_value=None, technology=None, region_code=None, subnational=None, xlim=None, gdp_shading=None):
        
        
        def thousands_format(x, pos):
            return f'{int(x):,}'

        # Convert the dataset into a dataframe
        supply_df = supply_ds.to_dataframe()
        
        # Merge with the country mapping
        merged_supply_df = pd.merge(supply_df, self.country_mapping.rename(columns={"index":"Country"}), how="left", on="Country")
        
        
        # Merge with country_mapping to give
        supply_df = merged_supply_df.copy().dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")

        # Create two copies
        uniform_df = merged_supply_df.copy().dropna(axis=0, subset=["Uniform_LCOE", "Country"], how="all")
        wacc_df = merged_supply_df.copy().dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")
        if subnational is not None:
            subnational_df = merged_supply_df.copy().dropna(axis=0, subset=["Subnational_LCOE", "Country"], how="all")
        
        # Convert units to TWh
        supply_df['technical_potential'] = supply_df['technical_potential'] / 1e+09
        uniform_df['technical_potential'] = uniform_df['technical_potential'] / 1e+09
        wacc_df['technical_potential'] = wacc_df['technical_potential'] / 1e+09
        if subnational is not None:
            subnational_df['technical_potential'] = subnational_df['technical_potential'] / 1e+09

        # For the Country WACC case, sort values and calculate cumulative sum
        supply_df = supply_df.round({'Estimated_WACC': 3})
        supply_df = supply_df.sort_values(by=['Estimated_WACC'], ascending=True)
        supply_df['cumulative_potential'] = supply_df['technical_potential'].cumsum()

        # For the WACC case, sort values and calculate cumulative sum
        wacc_sorted_df = wacc_df.sort_values(by=['Calculated_LCOE'], ascending=True)
        wacc_sorted_df['cumulative_wacc'] = wacc_sorted_df['technical_potential'].cumsum()

        # For the Uniform WACC case, sort values and calculate cumulative sum
        uniform_sorted_df = uniform_df.sort_values(by=['Uniform_LCOE'], ascending=True)
        uniform_sorted_df['cumulative_uniform'] = uniform_sorted_df['technical_potential'].cumsum()
        uniform_sorted_df = uniform_sorted_df.drop(uniform_sorted_df[uniform_sorted_df['cumulative_uniform'] > np.nanmax( supply_df['cumulative_potential'])].index)
        
        if subnational is not None:
            # For the Uniform WACC case, sort values and calculate cumulative sum
            subnational_sorted_df = subnational_df.sort_values(by=['Subnational_LCOE'], ascending=True)
            subnational_sorted_df['cumulative_uniform'] = subnational_sorted_df['technical_potential'].cumsum()
            subnational_sorted_df = subnational_sorted_df.drop(subnational_sorted_df[subnational_sorted_df['cumulative_uniform'] > np.nanmax( supply_df['cumulative_potential'])].index)
        
        # Print maximums
        print(f"The maximum for supply_df is {np.nanmax(supply_df['cumulative_potential'])}, the maximum for the uniform case is {np.nanmax(wacc_sorted_df['cumulative_wacc'])} and the max for country specific is {np.nanmax(uniform_sorted_df['cumulative_uniform'])}")
        
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

        # Plot the results
        fig, ax = plt.subplots(figsize=(20, 8))
        color_labels = {}
        cmap = mpl.colormaps['gnuplot_r']
        norm = mpl.colors.Normalize(vmin=0, vmax=50000, clip=True)  # Normalize to the range of GDP
        


        # Iterate through each data point and create a bar with the specified width
        for index, row in supply_df.iterrows():
            width = row['technical_potential']   # Bar width, in TWh
            height = row['Estimated_WACC']  # Bar height
            country = row['Country']
            region = row['Region']            # Cumulative production, in TWh
            cumulative_production = row['cumulative_potential'] 
            
            # Check for missing or invalid region values and set the default color
            if gdp_shading is None:
                region_shading = region_colors.get(region, "grey")  # Default to grey if region not found
                color = region_shading
            else:
                # Get GDP per capita 
                if np.isnan(country):
                    gdp_per_capita = np.nan
                else:
                    gdp_per_capita = self.GDP_country_mapping.loc[self.GDP_country_mapping['index'] == country, '2024'].values[0]
                if gdp_per_capita is None or gdp_per_capita=="no data" or np.isnan(float(gdp_per_capita)):
                    color = "gray"
                else:
                    color = cmap(norm(float(gdp_per_capita)))

                    

            # Plot a bar with the specified width, height, x-position, and color
            ax.bar(cumulative_production, height, width=-1 * width, align='edge', color=color)
        
        # Create the legend
        if gdp_shading is None:
            handles = [plt.Line2D([0], [0], color=color, lw=4, label=region) for region, color in region_colors.items()]
            ax.legend(handles=handles, title="Regions", loc="upper center", ncol=5, fontsize=15, title_fontsize=20)
        else:
            cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 450000,50000], format=',', extend="max", anchor=(1.0, 0.5), pad=-0.1)
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(thousands_format))
            cbar.set_label('GDP per capita (USDpp, 2024)', fontsize=20)
            cbar.ax.tick_params(labelsize=15)

        
        


        # Set labels
        ax.set_ylim(0, 30)
        ax.set_ylabel('WACC (%)', fontsize=20)
        ax.set_xlabel('Annual Electricity Potential (TWh/year)', fontsize=25)
        ax.set_title(title, fontsize=30)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_format))

        # Set the size of x and y-axis tick labels
        ax.tick_params(axis='x', labelsize=20)  # Adjust the labelsize as needed
        ax.tick_params(axis='y', labelsize=20)  # Adjust the labelsize as needed
        
        if xlim is not None:
            ax.set_xlim([0, xlim])
            if xlim > 500000:
                ax.xaxis.set_ticks(np.arange(0, xlim+200000, 200000))
            else:
                ax.xaxis.set_ticks(np.arange(0, xlim+5000, 5000))

        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
            
        if region_code is not None:
            ax.text(0.15, 0.9, technology + "\n" + region_code, transform=ax.transAxes, fontsize=20, fontweight='bold', ha="center", va="center")

        if filename is not None:
            if gdp_shading is not None:
                plt.savefig(filename + "_GDP.png", bbox_inches="tight")
            else:
                plt.savefig(filename + ".png", bbox_inches="tight")

        plt.show()

        return supply_df
    
    def get_concessionality_v3(calculated_data, required_wacc_data, concessional_rate, technology):
    
        # Convert data in netcdf to pandas dataframe
        merged_dataset = xr.Dataset({
        "Required_WACC": required_wacc_data,
        "index": self.country_grids['Country']
    })
        working_dataframe = merged_dataset.to_dataframe().reset_index()

        # Import the existing costs of commercial debt, equity and the modelled debt share
        if technology == "Wind":
            financing_values = calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Onshore_Wind_WACC_2023', 'Wind_Cost_Debt_2023', 'Wind_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Onshore_Wind_WACC_2023":"WACC_2023", "Wind_Cost_Debt_2023": "Debt_Cost_2023", "Wind_Cost_Equity_2023":"Equity_Cost_2023"})
        elif technology == "Solar":
            financing_values = calculated_data[['Country code', 'index', 'Debt_Share_2023', 'Solar_WACC_2023', 'Solar_Cost_Debt_2023', 'Solar_Cost_Equity_2023', 'Tax_Rate']].rename(columns={"Solar_WACC_2023":"WACC_2023", "Solar_Cost_Debt_2023": "Debt_Cost_2023", "Solar_Cost_Equity_2023":"Equity_Cost_2023"})
        financing_terms = pd.merge(working_dataframe, financing_values, how="left", on="index") 

        # Calculate debt-equity ratio
        financing_terms['Equity_Share_2023'] = (100 - financing_terms['Debt_Share_2023']) 
        financing_terms['Debt_Equity_Ratio'] =  financing_terms['Debt_Share_2023'] / financing_terms['Equity_Share_2023']

        # Set the cost of concessional financing 
        financing_terms['Concessional_Cost_2023'] = concessional_rate 


        # Calculate the share of concessional financing / commercial debt / commercial equity
        numerator =  (financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Required_WACC'] - financing_terms['Debt_Cost_2023'] * financing_terms['Debt_Equity_Ratio'] * (1 - financing_terms['Tax_Rate']/100) - financing_terms['Equity_Cost_2023']
        denominator = ((financing_terms['Debt_Equity_Ratio'] + 1) * financing_terms['Concessional_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100) - financing_terms['Equity_Cost_2023'] - financing_terms['Debt_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100) * financing_terms['Debt_Equity_Ratio'])
        financing_terms['Concessional_Debt_Share'] = numerator / denominator
        financing_terms['Commercial_Equity_Share'] = (1 - financing_terms['Concessional_Debt_Share']) / (financing_terms['Debt_Equity_Ratio'] + 1) 
        financing_terms['Commercial_Debt_Share'] = 1 - financing_terms['Concessional_Debt_Share'] - financing_terms['Commercial_Equity_Share'] 

        # Calculate shares of final
        financing_terms['Equity_Contribution'] = financing_terms['Commercial_Equity_Share'] * (financing_terms['Equity_Cost_2023'])
        financing_terms['Debt_Contribution'] = financing_terms['Commercial_Debt_Share'] * (financing_terms['Debt_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100) ) 
        financing_terms['Concessional_Contribution'] = financing_terms['Concessional_Debt_Share'] * financing_terms['Concessional_Cost_2023'] * (1 - financing_terms['Tax_Rate']/100)
        financing_terms['Concessionality'] = financing_terms['Debt_Cost_2023'] - financing_terms['Concessional_Cost_2023']

        # Create a check
        financing_terms['Total_Check'] =  financing_terms['Concessional_Debt_Share'] + financing_terms['Commercial_Debt_Share'] + financing_terms['Commercial_Equity_Share']
        financing_terms['WACC_Concessional'] =  financing_terms['Equity_Contribution'] + financing_terms['Debt_Contribution'] + financing_terms['Concessional_Contribution']

        # Address inequalities
        financing_terms['Concessional_Debt_Share'] = financing_terms['Concessional_Debt_Share'] * 100
        financing_terms.loc[financing_terms['Required_WACC'] == 999, "Concessional_Debt_Share"] = 999
        financing_terms.loc[financing_terms['Required_WACC'] == 111, "Concessional_Debt_Share"] = 111
        financing_terms.loc[financing_terms['Required_WACC'] < concessional_rate, "Concessional_Debt_Share"] = 999
        financing_terms.loc[financing_terms['Required_WACC'] < concessional_rate, "Commercial_Equity_Share"] = 999
        financing_terms.loc[financing_terms['Required_WACC'] < concessional_rate, "Commercial_Debt_Share"] = 999

        # Convert back to netcdf
        financing_terms = financing_terms.set_index(["latitude", "longitude"])
        processed_data = financing_terms.to_xarray()

        return financing_terms, processed_data 
    
    
    
    def plot_TIAM_region(self, region_code, solar_data, wind_data, regional_solar_wacc, regional_wind_wacc):
        
        # Get Solar and Wind Datasets with technical potential
        solar_ds = self.get_supply_curves_v2(solar_data,  "Solar")
        wind_ds = self.get_supply_curves_v2(wind_data, "Wind")        
        
        # Plot region values
        wind_df = self.produce_wacc_potential_curve_v2(wind_ds, uniform_value=regional_wind_wacc, region_code=region_code, technology="Onshore\nWind")
        solar_df = self.produce_wacc_potential_curve_v2(solar_ds, uniform_value=regional_solar_wacc, region_code=region_code, technology="Solar")
           
    
    
        # create the heatmap using pcolormesh
        if anchor is None:
            anchor = 0.355
        fig = plt.figure(figsize=(30, 15), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        if center_norm is None:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.Normalize(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
        else:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.SymLogNorm(vmin = tick_values[0], vmax=tick_values[-1], linscale
    =1, linthresh=1), transform=ccrs.PlateCarree(), cmap=cmap)

        # Check if there is a need for extension
        values_min = np.nanmin(values)
        values_max = np.nanmax(values)
        if values_min < tick_values[0]:
            extend = "min"
        elif values_max > tick_values[-1]:
            extend = "max"
        elif (values_max > tick_values[-1]) & (values_min < tick_values[0]):
            extend = "both"
        else:
            extend="neither"
        if extend_set is not None:
            extend = extend_set
        

        axins = inset_axes(
        ax,
        width="1.5%",  
        height="80%",  
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
        cb = fig.colorbar(heatmap, cax=axins, shrink=0.5, ticks=tick_values, format="%0.0f", extend=extend, anchor=(0, anchor))


        cb.ax.tick_params(labelsize=20)
        if title is not None:
            cb.ax.set_title(title, fontsize=25)

        # Add the special shading
        if special_value is not None:
            special_overlay = np.where(values == special_value, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())

        if special_value_2 is not None:
            special_overlay = np.where(values == special_value_2, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['\\'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

        # set the extent and aspect ratio of the plot
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
        ax.set_aspect(1)

        # add axis labels and a title
        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
        ax.add_feature(borders, edgecolor='gray', linestyle=':')
        ax.coastlines()
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        ax.coastlines()
        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        hatch_patches=[]
        if special_value is not None and hatch_label is not None:
            hatch_patch_1 = Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label)
            hatch_patches.append(hatch_patch_1)

        if hatch_label_2 is not None:
            hatch_patch_2 = Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2)
            hatch_patches.append(hatch_patch_2)

        if hatch_patches:
            ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        return 
    
    def get_wacc_values(self, estimated_waccs, country_mapping, technology): 

        data = country_mapping
        storage_df = xr.zeros_like(data['land'])
        if technology == "Offshore Wind":
            storage_df = xr.zeros_like(data['sea'])
        storage_df = xr.where(storage_df == 0, np.nan, np.nan)
        for i in np.arange(1, 251, 1):
            # Extract WACC
            if technology == "Solar":
                wacc = estimated_waccs[estimated_waccs['index'] == i]['solar_pv_wacc'].values[0] 
            elif technology == "Onshore Wind":
                wacc = estimated_waccs[estimated_waccs['index'] == i]['onshore_wacc'].values[0] 
            elif technology == "Offshore Wind":
                wacc = estimated_waccs[estimated_waccs['index'] == i]['offshore_wacc'].values[0] 

            # Apply mapping
            storage_df = xr.where(data['land'] == i, wacc, storage_df)

        # Extracted data
        extracted_data = storage_df

        return extracted_data
    
    def plot_wacc_values(self, geodata):
        
        # Reindex geodata
        country_geodata = geodata.reindex({"latitude":self.solar_results.latitude, "longitude":self.solar_results.longitude}, method="nearest")
        
        # Get solar WACCs
        solar_plot_waccs = self.get_wacc_values(self.country_wacc_mapping, country_geodata, "Solar")
        solar_plot_waccs = xr.where(np.isnan(self.solar_results['Calculated_LCOE']), np.nan, solar_plot_waccs)
        
        # Get onshore WACCs
        onshore_plot_waccs = self.get_wacc_values(self.country_wacc_mapping, country_geodata, "Onshore Wind")
        onshore_plot_waccs = xr.where(np.isnan(self.solar_results['Calculated_LCOE']), np.nan, onshore_plot_waccs)
        
        # Get offshore WACCs
        offshore_plot_waccs = self.get_wacc_values(self.country_wacc_mapping, country_geodata, "Offshore Wind")
        offshore_plot_waccs = xr.where(np.isnan(onshore_plot_waccs), np.nan, offshore_plot_waccs)
        
        # Get collation
        collated_wind_results = self.collated_wind_results
        solar_results = self.solar_results
        
        # Plot data
        #elf.plot_data_shading(solar_plot_waccs,solar_plot_waccs.latitude, solar_plot_waccs.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC \n (%, nom,\nafter tax)\n", cmap="YlOrRd", extend_set="neither", filename = self.output_folder + "Solar_WACC_2023", graphmarking="a")
        #elf.plot_data_shading(onshore_plot_waccs, onshore_plot_waccs.latitude, onshore_plot_waccs.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC\n (%, nom,\nafter tax)\n", cmap="YlGnBu", extend_set="neither", filename = self.output_folder + "Onshore_Wind_WACC_2023", graphmarking="b")
        #elf.plot_data_shading(offshore_plot_waccs, offshore_plot_waccs.latitude, offshore_plot_waccs.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC\n (%, nom,\nafter tax)\n", cmap="YlGnBu", extend_set="neither", filename = self.output_folder + "Offshore_Wind_WACC_2023", graphmarking="c")
        self.plot_data_shading(solar_results['Estimated_WACC'], solar_results.latitude, solar_results.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC\n (%, nom,\nafter tax)\n", cmap="YlOrRd", extend_set="neither", filename = self.output_folder + "Solar_WACC_2023", graphmarking="a")
        self.plot_data_shading(collated_wind_results['Estimated_WACC'], collated_wind_results.latitude, collated_wind_results.longitude, tick_values = [0, 5, 10, 15, 20, 25], title="Estimated\nWACC\n (%, nom,\nafter tax)\n", cmap="YlGnBu", extend_set="neither", filename = self.output_folder + "Wind_WACC_2023", graphmarking="b")
    
    
    def plot_LCOE_comparison(self):
        
        # Drop latitude = 0
        wind_results = self.wind_results.drop_sel(latitude=0)
        solar_results = self.solar_results
        offshore_results = self.offshore_results.drop_sel(latitude=0)
        collated_wind_results = self.collated_wind_results
        
        # Plot corresponding graphs for solar LCOE
        self.plot_data_shading(solar_results['Calculated_LCOE'], solar_results.latitude, solar_results.longitude, tick_values=[0, 25, 50, 75, 100, 150], graphmarking="a", cmap="YlOrRd", filename = self.output_folder + "Solar_LCOE_Country", title="LCOE\n (US$/MWh)\n")
        self.plot_data_shading(solar_results['Uniform_LCOE'], solar_results.latitude, solar_results.longitude, tick_values=[0, 25, 50, 75, 100, 150], graphmarking="b", cmap="YlOrRd", filename = self.output_folder + "Solar_LCOE_Uniform", title="LCOE\n (US$/MWh)\n")
        #self.plot_data_shading(solar_results['Reduced_LCOE'], solar_results.latitude, solar_results.longitude, tick_values=[0, 25, 50, 75, 100, 150], graphmarking="b", cmap="YlGnBu", filename = self.output_folder + "Solar_LCOE_Reduced", title="LCOE\n (US$/MWh)\n")
        
        # Plot graphs for share of financing costs for solar
        self.plot_data_shading(solar_results['WACC_LCOE_Country'], solar_results.latitude, solar_results.longitude, tick_values=[0, 25, 50, 75, 100], graphmarking="a", cmap="YlOrRd", filename = self.output_folder + "Solar_LCOE_Financing_Share_Country", title="LCOE share\n attributable to\nfinance (%)\n")
        self.plot_data_shading(solar_results['WACC_LCOE_Uniform'], solar_results.latitude, solar_results.longitude, tick_values=[0, 25, 50, 75, 100], graphmarking="a", cmap="YlOrRd", filename = self.output_folder + "Solar_LCOE_Financing_Share_Uniform", title="LCOE share\n attributable to\nfinance (%)\n")
        
        # Plot corresponding graphs for wind LCOE
        
        # Plot all wind results
        self.plot_data_shading(collated_wind_results['Calculated_LCOE'], collated_wind_results.latitude, collated_wind_results.longitude, tick_values=[0, 25, 50, 75, 100, 150, 250], graphmarking="a", cmap="YlGnBu", filename = self.output_folder + "Wind_LCOE_Country", title="LCOE\n (US$/MWh)\n")
        self.plot_data_shading(collated_wind_results['Uniform_LCOE'], collated_wind_results.latitude, collated_wind_results.longitude, tick_values=[0, 25, 50, 75, 100, 150, 250], graphmarking="b", cmap="YlGnBu", filename = self.output_folder + "Wind_LCOE_Uniform", title="LCOE\n (US$/MWh)\n")
        
        # Plot shares of financing costs for wind
        self.plot_data_shading(collated_wind_results['WACC_LCOE_Country'], collated_wind_results.latitude, collated_wind_results.longitude, tick_values=[0, 25, 50, 75, 100], graphmarking="b", cmap="YlGnBu", filename = self.output_folder + "Wind_LCOE_Financing_Share_Country", title="LCOE share\n attributable to\nfinance (%)\n")
        self.plot_data_shading(collated_wind_results['WACC_LCOE_Uniform'], collated_wind_results.latitude, collated_wind_results.longitude, tick_values=[0, 25, 50, 75, 100], graphmarking="b", cmap="YlGnBu", filename = self.output_folder + "Wind_LCOE_Financing_Share_Uniform", title="LCOE share\n attributable to\nfinance (%)\n")
        
        
        # Plot change in LCOE for wind and solar
        #elf.plot_data_shading((collated_wind_results['Calculated_LCOE'] - collated_wind_results['Uniform_LCOE']), collated_wind_results.latitude, collated_wind_results.longitude, tick_values=[-50, -25, 0, 25, 50, 100], graphmarking="b", cmap="coolwarm", filename = self.output_folder + "Wind_LCOE_Change", title="Wind:\nIncrease in\nLCOE from\n4.8% Scenario\n(US$/MWh)\n")
    
        # Plot change in LCOE for wind and solar from reduced condition
        self.plot_data_shading(((solar_results['Calculated_LCOE']/solar_results['Uniform_LCOE']) -1)*100, solar_results.latitude, solar_results.longitude, tick_values=[-25, -20, -15, -10, -5,  0, 25, 50, 75, 100, 150], graphmarking="c", cmap="RdBu_r", filename = self.output_folder + "Solar_LCOE_PercentChange", title="Solar PV:\nIncrease\nin LCOE\n(%)\n", center_norm=0)
        self.plot_data_shading(((collated_wind_results['Calculated_LCOE']/collated_wind_results['Uniform_LCOE']) -1)*100, collated_wind_results.latitude, collated_wind_results.longitude, tick_values=[-25, -20, -15, -10, -5, 0, 25, 50, 75, 100, 150], graphmarking="d", cmap="RdBu_r", filename = self.output_folder + "Wind_LCOE_PercentChange", title="Wind:\nIncrease\nin LCOE\n(%)\n", center_norm=0)
        
        # Plot increase in LCOE for wind and solar from reduced condition
        self.plot_data_shading((solar_results['Calculated_LCOE'] - solar_results['Uniform_LCOE']), solar_results.latitude, solar_results.longitude, tick_values=[-25, -20, -15, -10, -5,  0, 25, 50, 75, 100], graphmarking="a", cmap="bwr", filename = self.output_folder + "Solar_LCOE_Change", title="Solar PV:\n Increase\nin LCOE\n(US$/MWh)\n", center_norm=0)
        self.plot_data_shading((collated_wind_results['Calculated_LCOE'] - collated_wind_results['Uniform_LCOE']), collated_wind_results.latitude, collated_wind_results.longitude, tick_values=[-25, -20, -15, -10, -5,  0, 0, 25, 50, 75, 100, 150], graphmarking="b", cmap="bwr", filename = self.output_folder + "Wind_LCOE_Change", title="Wind:\n Increase\nin LCOE\n(US$/MWh)\n", center_norm=0)
    
    
    def plot_supply_curve_global(self, solar_uniform, wind_uniform, offshore_uniform, subnational=None, gdp_shading=None):
        
        # Get Solar and Wind Datasets with technical potential
        solar_ds = self.get_supply_curves_v2(self.solar_results,  "Solar")
        wind_ds = self.get_supply_curves_v2(self.wind_results, "Onshore Wind")
        offshore_ds = self.get_supply_curves_v2(self.offshore_results, "Offshore Wind")
        
        # Plot solar and wind wacc potential curves
        solar_df = self.produce_potential_curve_v3(solar_ds, uniform_value=solar_uniform, technology="Solar", region_code="Global", graphmarking="a", filename=self.output_folder + "/Solar_Global", subnational=subnational, xlim=1e+06, gdp_shading=gdp_shading)
        wind_df = self.produce_potential_curve_v3(wind_ds, uniform_value=wind_uniform, technology="Onshore\nWind", region_code="Global", graphmarking="b",  filename=self.output_folder + "/Onshore_Wind_Global", subnational=subnational, xlim=1e+06, gdp_shading=gdp_shading)
        offshore_df = self.produce_potential_curve_v3(offshore_ds, uniform_value=offshore_uniform, technology="Offshore\nWind", region_code="Global", graphmarking="c",  filename=self.output_folder + "/Offshore_Wind_Global", subnational=subnational, xlim=1.5e+04, gdp_shading=gdp_shading)
        
        # Store dataframes
        self.solar_df = solar_df
        self.wind_df = wind_df
        
        
    def plot_supply_curves(self, results, country_index):
    
        # Select country
        national_results = xr.where(self.land_grids == country_index, results, np.nan)

        # Convert into dataframe
        national_df = national_results.to_dataframe()

        # Create axis
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")

        # Color 
        color = ["red", "blue", "black"]

        # Sort each of the three variables
        for i, variable in enumerate(["Calculated_LCOE", "Uniform_LCOE", "Subnational_LCOE"]):

            # Extract data
            extracted_data = national_df[[variable, "electricity_production"]]

            # Sort values
            extracted_data = extracted_data.sort_values(by=[variable], ascending=True)

            # Create cumulative sum
            extracted_data['cumulative_production'] = extracted_data['electricity_production'].cumsum()

            # Plot
            ax.plot(extracted_data['cumulative_production'], extracted_data[variable], color=color[i], lw=2.5, label=variable, linestyle="--")

        # Set axis limits
        ax.set_ylim([0, 100])
        ax.text(0.02, 0.9, country_index, transform=ax.transAxes, fontsize=15, fontweight='bold')
        
    def run_TIAM_regions(self, solar_uf, onshore_uf, offshore_uf):

        # Extract the relevant data
        TIAM_regions = self.TIAM_regions 

        # Specify input
        solar_results = self.solar_results
        onshore_results = self.wind_results
        offshore_results = self.offshore_results

        # Call postprocessing function to create the corresponding supply curve
        self.plot_TIAM_lcoe(solar_results, onshore_results, offshore_results, solar_uf, onshore_uf, offshore_uf)

        

        
    def plot_TIAM_lcoe(self, solar_filtered_results, onshore_filtered_results, offshore_filtered_results, solar_uf, onshore_uf, offshore_uf):

        # Get Solar and Wind Datasets with technical potential
        solar_ds = self.get_supply_curves_v2(solar_filtered_results,  "Solar", LCOE_cutoff="True")
        wind_ds = self.get_supply_curves_v2(onshore_filtered_results, "Onshore Wind", LCOE_cutoff="True")
        offshore_ds = self.get_supply_curves_v2(offshore_filtered_results, "Offshore Wind", LCOE_cutoff="True")


        # Plot corresponding results separately
        self.produce_lcoe_potential_v1(solar_ds, xlim=250000, graphmarking="a", uniform_value=solar_uf/100, position="upper", filename="LCOE_Supply_Solar", technology="Solar")
        self.produce_lcoe_potential_v1(wind_ds, xlim=250000, graphmarking="b", uniform_value=onshore_uf/100, position="upper", filename="LCOE_Supply_Onshore", technology="Onshore Wind")
        self.produce_lcoe_potential_v1(offshore_ds, xlim=2500, technology="Offshore Wind", graphmarking="c", uniform_value=offshore_uf/100, position="lower", filename="LCOE_Supply_Offshore")


    def produce_lcoe_potential_v1(self, supply_ds, filename=None, graphmarking=None, position=None, uniform_value=None, technology=None, region_code=None, subnational=None, xlim=None):

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

            # Print the results
            print(f"{region}:Cumulative potential at LCOE {target_lcoe} for country WACC: {country_cumulative_at_target} TWh, {country_cumulative_at_target/country_cumulative_max*100:0.2f}% of {country_cumulative_max}TWh")
            print(f"{region}:Cumulative potential at LCOE {target_lcoe} for uniform WACC: {uniform_cumulative_at_target} TWh, {uniform_cumulative_at_target/uniform_cumulative_max*100:0.2f}% of {uniform_cumulative_max}TWh")

        # Convert the dataset into a dataframe
        supply_df = supply_ds.to_dataframe()

        # Merge with the country and region mapping
        merged_supply_df = pd.merge(supply_df, self.country_mapping.rename(columns={"index":"Country"}), how="left", on="Country")

        # Merge with country_mapping to give
        supply_df = merged_supply_df.copy().dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")

        # Create two copies
        uniform_df = merged_supply_df.copy().dropna(axis=0, subset=["Uniform_LCOE", "Country"], how="all")
        wacc_df = merged_supply_df.copy().dropna(axis=0, subset=["Calculated_LCOE", "Country"], how="all")
        if subnational is not None:
            subnational_df = merged_supply_df.copy().dropna(axis=0, subset=["Subnational_LCOE", "Country"], how="all")

        # Convert units to TWh
        supply_df['technical_potential'] = supply_df['technical_potential'] / 1e+09
        uniform_df['technical_potential'] = uniform_df['technical_potential'] / 1e+09
        wacc_df['technical_potential'] = wacc_df['technical_potential'] / 1e+09
        if subnational is not None:
            subnational_df['technical_potential'] = subnational_df['technical_potential'] / 1e+09

        # For the WACC case, sort values and calculate cumulative sum
        wacc_sorted_df = wacc_df.sort_values(by=['Region', 'Calculated_LCOE'], ascending=True)
        wacc_sorted_df['cumulative_potential'] = wacc_sorted_df.groupby('Region')['technical_potential'].cumsum()
        wacc_grouped = wacc_sorted_df.groupby('Region')

        # For the Uniform WACC case, sort values and calculate cumulative sum
        uniform_sorted_df = uniform_df.sort_values(by=['Region', 'Uniform_LCOE'], ascending=True)
        uniform_sorted_df['cumulative_potential'] = uniform_sorted_df.groupby('Region')['technical_potential'].cumsum()
        uniform_grouped = uniform_sorted_df.groupby('Region')

        if subnational is not None:
            # For the Uniform WACC case, sort values and calculate cumulative sum
            subnational_sorted_df = subnational_df.sort_values(by=['Region', 'Uniform_LCOE'], ascending=True)
            subnational_sorted_df['cumulative_potential'] = subnational_sorted_df.groupby('Region')['technical_potential'].cumsum()
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
            target_lcoe = 55
        elif technology == "Offshore Wind":
            target_lcoe = 106
        else:
            target_lcoe = 73
                               
        if technology == "Solar":
            min_lcoe = 29
            average_lcoe = 55 
            max_lcoe = 92
        elif technology == "Offshore Wind":
            min_lcoe = 74
            average_lcoe = 106
            max_lcoe = 139
        else:
            min_lcoe = 27
            average_lcoe = 50
            max_lcoe = 73
        
        # Plot the results
        if technology == "Offshore Wind":
            width_ratio = [1, 1]
        else:
            width_ratio = [2, 1]
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

            # Produce plots of LCOE against supply for each region
            ax1.plot(country_wacc_potential,country_wacc_lcoe , color=region_shading, linestyle="-")
            ax1.plot(uniform_wacc_potential,uniform_wacc_lcoe, color=region_shading, linestyle="--")

            # Plot the subnational if applicable
            if subnational is not None:
                subnational_wacc_potential = subnational_grouped[['cumulative_potential']].get_group(region)
                subnational_wacc_lcoe = subnational_grouped[['Subnational_LCOE']].get_group(region)
                ax1.plot(subnational_wacc_potential,subnational_wacc_lcoe, color=region_shading, linestyle=":")

        # Create the legend
        handles = [plt.Line2D([0], [0], color=color, lw=4, label=region) for region, color in region_colors.items()]
        ax1.legend(handles=handles, title="Regions", loc=position+ " right", ncol=5, fontsize=12, title_fontsize=15)

        # Set labels
        ax1.set_ylim(0, 250)
        ax1.set_xlim(0, xlim)
        ax1.set_ylabel('Levelised Cost of Electricity\n(USD/MWh, '+ technology + ')', fontsize=18)
        ax1.set_xlabel('Developing Regions\nAnnual Electricity Potential (TWh/year)', fontsize=18)
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
        ax2.legend(handles=style_handles, title="Cost of Capital", loc=position+ " left", ncol=1, fontsize=12, title_fontsize=15)
            
        # Add shading
        ax1.axhspan(min_lcoe, average_lcoe, color="lightblue", alpha=0.1)
        #ax1.axhspan(average_lcoe, max_lcoe, color="lightgreen", alpha=0.1)
        ax2.axhspan(min_lcoe, average_lcoe, color="lightblue", alpha=0.1)
        #ax2.axhspan(average_lcoe, max_lcoe, color="lightgreen", alpha=0.1)
        ax1.text(0.85*xlim, (min_lcoe+average_lcoe)/2, "Lazard lower\nLCOE range", fontsize=10, ha="center", va="center")
        #ax1.text(0.85*xlim, (max_lcoe+average_lcoe)/2, "US upper\nLCOE range", fontsize=10, ha="center", va="center")

        # Set labels
        ax2.set_ylim(0, 250)
        if technology == "Offshore Wind":
            ax2.set_xlim(0, xlim)
        else:
            ax2.set_xlim(0, xlim/2)

        ax2.set_xlabel('Developed & Industrialising Regions\nAnnual Electricity Potential (TWh/year)', fontsize=18)
        ax2.xaxis.set_major_formatter(FuncFormatter(thousands_format))

        # Set the size of x and y-axis tick labels
        ax2.tick_params(axis='x', labelsize=15)  # Adjust the labelsize as needed
        ax2.tick_params(axis='y', labelsize=0)  # Adjust the labelsize as needed

        if xlim is not None:
            if technology == "Offshore Wind":
                ax1.xaxis.set_ticks(np.arange(0, xlim, 500))
                ax2.xaxis.set_ticks(np.arange(0, (xlim), 500))
            else:
                ax1.xaxis.set_ticks(np.arange(0, xlim, 25000))
                ax2.xaxis.set_ticks(np.arange(0, (xlim/2), 25000))

        if graphmarking is not None:
            ax1.text(0.02, 0.94, graphmarking, transform=ax1.transAxes, fontsize=20, fontweight='bold')

        if region_code is not None:
            ax1.text(0.15, 0.9, technology + "\n" + region_code, transform=ax1.transAxes, fontsize=20, fontweight='bold', ha="center", va="center")

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        plt.show()
        
        
    def plot_data_shading(self, values, latitudes, longitudes, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend_set=None, graphmarking=None, special_value=None, hatch_label=None, hatch_label_2=None, special_value_2=None, center_norm=None):      
    
        # create the heatmap using pcolormesh
        if anchor is None:
            anchor = 0.355
        fig = plt.figure(figsize=(30, 15), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        if center_norm is None:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.Normalize(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
        else:
            heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.TwoSlopeNorm(vmin=tick_values[0], vcenter=center_norm, vmax=tick_values[-1]), cmap=cmap)

        # Check if there is a need for extension
        values_min = np.nanmin(values)
        values_max = np.nanmax(values)
        if values_min < tick_values[0]:
            extend = "min"
        elif values_max > tick_values[-1]:
            extend = "max"
        else:
            extend = "neither"
        if extend_set is not None:
            extend = extend_set
        if (values_max > tick_values[-1]) & (values_min < tick_values[0]):
            extend = "both" 
           

        axins = inset_axes(
        ax,
        width="1.5%",  
        height="80%",  
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
        cb = fig.colorbar(heatmap, cax=axins, shrink=0.5, ticks=tick_values, format="%0.0f", extend=extend, anchor=(0, anchor))


        cb.ax.tick_params(labelsize=20)
        if title is not None:
            cb.ax.set_title(title, fontsize=25)

        # Add the special shading
        if special_value is not None:
            special_overlay = np.where(values == special_value, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())

        if special_value_2 is not None:
            special_overlay = np.where(values == special_value_2, 1, np.nan)
            hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

        # set the extent and aspect ratio of the plot
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
        ax.set_aspect(1)

        # add axis labels and a title
        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
        ax.add_feature(borders, edgecolor='gray', linestyle=':')
        ax.coastlines()
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        ax.coastlines()
        if graphmarking is not None:
            ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')

        hatch_patches=[]
        if special_value is not None and hatch_label is not None:
            hatch_patch_1 = Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label)
            hatch_patches.append(hatch_patch_1)

        if hatch_label_2 is not None:
            hatch_patch_2 = Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2)
            hatch_patches.append(hatch_patch_2)

        if hatch_patches:
            ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)

        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches="tight")

        return 

    def calculate_total_cost(self, supply_curve_ds, fraction):
    
        # Convert to a dataframe
        supply_df = supply_curve_ds.to_dataframe()

        # Calculate the increase in cost at each location
        supply_df["WACC_COST"] = supply_df["technical_potential"]/1000 * fraction * np.where(supply_df["Calculated_LCOE"] - supply_df["Uniform_LCOE"] > 0, supply_df["Calculated_LCOE"] - supply_df["Uniform_LCOE"], 0)

        # Sum up all costs
        total_increase = supply_df["WACC_COST"].agg("sum") / 1e+09

        # Print the value
        print(f"Total increase due to the influence of WACC is {total_increase:0.2f} billion US")

 