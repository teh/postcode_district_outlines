# Warning

The code in its current state is me sketching and pretty much unusable for production.

I will clean up the code when I am happy with the results.

# The problem

The postcode data released by the OS in Great Britain are coded as points, not as areas.
To generate pretty heat-maps you need to colour areas instead of points though.

This project aims to generate a closed outline for each postal district.

The ideal outlines would cover all of the land. That implies that two adjacent postcode outlines share boundary points where appropriate.

# How it is done in theory

1. Create a grid
2. Use K nearest neighbours (KNN) to figure out what district each grid point belongs to
3. Use marching squares to outline each district
4. Clip resulting outlines against the UK coastline

# How it is done in practice

1. Chose roughly the mid-point (x0, y0) for each district
2. Use a "lazy" variant of marching squares starting at (x0, y0). The lazy variant follows the outline instead of running KNN on the entire grid. This is necessary because at interesting resolutions the amount of data (n*m) is in the terabyte range. That is too much for my laptop.
3. Clip the resulting outline against 

[1]
The Ordnance Survey (OS) made a lot of their data freely available in 2011. This is
really awesome stuff.

Unfortunately a different agency maps Northern Ireland, so this data doesn't
contain any BT postcodes.