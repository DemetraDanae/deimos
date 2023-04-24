# deimos
deimos, is a computational methodology for optimal grouping, applied on the read-across prediction of engineered nanomaterials’ (NMs) toxicity-related properties. The method is based on the formulation and the solution of a mixed-integer optimization program (MILP) problem that automatically and simultaneously
performs feature selection, defines the grouping boundaries according to the response variable and develops linear regression models in each group. For each group/region, the characteristic centroid is defined in order to allocate untested NMs to the groups. The deimos MILP problem is integrated in a broader optimization workflow that selects the best performing methodology between the standard multiple linear regression (MLR), least absolute shrinkage and selection operator (LASSO) models and the proposed deimos multiple-region model. This method can be applied to property prediction of other than NM chemical entities and it is not limited to NMs toxicity prediction.

<!-- The relevant publication "deimos: a novel automated methodology for optimal grouping. Application to nanoinformatics case studies." has been published at Wiley's Molecular Informatics and can be found <a href="">here</a>. -->

# Datasets
The deimos methodology is demonstrated on two publicly available datasets: 
<ul>
<li>The first one (Gold ENMs), presented by Walkey <i>et al</i>. (<a href="https://doi.org/10.1021/nn406018q">2014</a>) and filtered by Varsou <i>et al</i>. (<a href="https://doi.org/10.1021/acs.jcim.7b00160">2017</a>) using toxFlow. 
<li>The second dataset (MeOx ENMs) is included in the study of Forest <i>et al</i>. (<a href="https://doi.org/10.1007/s11051-019-4541-2">2019</a>).
</ul>

# License
This application is released under <a href="https://www.gnu.org/licenses/gpl.html"> GNU General Public License v.3</a>.
```html

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
You should have received a copy of the GNU General Public License along with this program.  
If not, see here: http://www.gnu.org/licenses/.
