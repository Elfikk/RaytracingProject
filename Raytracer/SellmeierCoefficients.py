from numpy import array

#Sellmeier approximation holds from near UV to 2.3 micro m, as per Schlott, so
#it's well within our range.

#From the random paper I found 
#Only two coefficients were given for water - likely due to its pretty
#steady refractive index profile (1.33 through entire visible spectrum).
water_Bs = array([0.75831, 0.08495]) 
water_Cs = array([0.01007, 8.91377])

#F2 in the Schlott Data Sheet
#Type of 'Flint Glass' - high dispersion glass 
flint_glass_Bs = array([1.34533359, 0.209073176, 0.937357162]) 
flint_glass_Cs = array([0.00997743871, 0.0470450767, 111.8867640])

#K7 in the Schlott Data Sheet
#Type of Crown (Krone in German, hence K) - low dispersion glass. Often
#combined with F2 to make an achromatic doublet (ty wikipedia).
crown_glass_Bs = array([1.12735550, 0.124412303, 0.827100531])
crown_glass_Cs = array([0.00720341707, 0.0269835916, 100.3845880])