import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.table import Table, vstack
from scipy.cluster.vq import kmeans, vq
import os
from otherFun import match_table_lengths
from jackknifing import jackknife_weighted_mean_cov_fast2

# resulting plots and data can be found in directory boxed_script/{name}
def boxey_feedback(table, n_boxes, frequency, filterType, weights='uniform', splitMethod='sfrM_median', splitProp=False, boxPlot=False, name='test', preboxed=False, presplit= False):
    '''
    Input:
    
        table (astropy.table): astropy table that has property cuts made to it already. 
        n_boxes (int): number of boxes for the data to split into on the redshift and mass distribution
        frequency (string): 150ghz, 90ghz, kappa, compy
        filterType (string): ringring2, ring
        weights (string): 
            'uniform': uniform weighting system; weights are defined by np.ones
            'mass': mass weighting system; more massive objects have more weight than less massive objects
        splitMethod (string): each box is split into high and low feedback via their log(m) by log(sfr/m) distributions
            'sfrM_fit': best fit line made by log(m) distribution, then the split is defined by positive or negative difference from the best fit line
            'sfrM_median': horizontal split based on the median values of log(sfr/m)
            'aglum': split based on median of agnlum, high and low values.
            'prop': median of a particular property
        splitProp (string): property to split on via median (must be written the same as it is in the table)
        boxPlot (bool): print what the boxed redshift v mass distribution
        preboxed (bool): if your table already contains boxes, you can label this as true so the run skips the boxing procedure
        presplit (bool): if your table already contains splits within boxes + boxes column, you can label this as true so the run skips the splitting procedure
        name (string): output name

    Output:

        covariance, chi2
    '''

    ### check if the input are allowed inputs
    weights_allowed = ['uniform', 'mass']
    splitMethod_allowed = ['sfrM_fit', 'sfrM_median', 'prop']
    
    if weights not in weights_allowed:
        raise ValueError(f"Invalid weights input: '{weights}'. Must be one of {weights_allowed}.")

    if splitMethod not in splitMethod_allowed:
        raise ValueError(f"Invalid split_method input: '{splitMethod}'. Must be one of {splitMethod_allowed}.")

    if type(n_boxes) is not int:
        raise ValueError(f"Invalid n_boxes input type: '{n_boxes}'. Must be an integer.")

    if (splitMethod == 'prop') and (splitProp is False):
        raise ValueError(f"Please indicate a property to split on.")

    ###

    os.makedirs(f'boxed_script/{name}', exist_ok=True)

    if weights == 'mass':
        calculated_weights = 10**table['LOGM'] / np.sum(10**table['LOGM'])
        table['mass_weights'] = calculated_weights

    table = table[table[f'catmask_{filterType}_{frequency}']==True]

    if preboxed == False:
        '''
        Splitting
        '''
        data = np.vstack((table['Z'], table['LOGM'])).T
        
        # k-means to group into n_boxes clusters
    
        print(f'making boxes')
        centroids, _ = kmeans(data, n_boxes)
        labels, _ = vq(data, centroids)
        
        boxes = {}
        for i in range(n_boxes):
            mask = labels == i
            box_data = data[mask]
            boxes[i] = {
                'z': box_data[:, 0],
                'logm': box_data[:, 1],
                'center': centroids[i]
        }
    
        if boxPlot:
            fig, ax = plt.subplots()
            scatter = ax.scatter(table['Z'], table['LOGM'], c=labels, cmap=f'tab20', s=0.1)
            
            # Show box centers and draw boundaries (approximate as circles)
            for i in range(n_boxes):
                cx, cy = boxes[i]['center']
                ax.plot(cx, cy, 'kx', markersize=12)
                ax.text(cx, cy, f'Box {i}', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6))
            
            ax.set_title('Scatter Plot with Clustered Boxes')
            ax.set_xlabel('Z')
            ax.set_ylabel('LOGM')
            plt.grid(True)
            plt.savefig(f'boxed_script/{name}/boxey_plot.png', bbox_inches='tight')
            plt.close()
    
    '''
    Save box number to table and split into high and low feedback groups for every box
    '''

    if preboxed:
        boxes = np.array(np.unique(table['box']))

    elif presplit == False:
        completed=Table()
        completed.write(f'boxed_script/{name}/{name}_boxes.fits', overwrite=True) # keeps all objects in the table
        completed.write(f'boxed_script/{name}/{name}_boxes_evenSplit.fits', overwrite=True) # throws out a few objects (maximum = n_boxes) to make the high and low bins even as jackknifing only works with even bins.
        for i in range(len(boxes)):
            if preboxed == False:
                mask = list(zip(table['Z'], table['LOGM']))
                targets = set(zip(boxes[i]['z'], boxes[i]['logm']))
                
                matched = table[[pair in targets for pair in mask]]
                matched['box']=i
                dd = matched[matched['LOGM'].argsort()]
    
            elif preboxed == True:
                thisbox = table[table['box']==i]
                dd = thisbox[thisbox['LOGM'].argsort()]
    
            if splitMethod == 'sfrM_fit':
                v=dd['LOGSFR']-dd['LOGM']
                dd['sfr/m']=v
                A = np.polyfit(dd['LOGM'], v, 2)
                pp = np.poly1d(A)
                ff=v-pp(dd['LOGM'])
                
                dd['fdb']=ff
                
                mea = np.median(dd['fdb'])
                lo=dd[dd['fdb']>=mea] #low
                hi=dd[dd['fdb']<mea] #high
    
            if splitMethod == 'sfrM_median':
                v=dd['LOGSFR']-dd['LOGM']
                dd['sfr/m']=v
                mea = np.median(10**dd['sfr/m'])
                lo=dd[dd['sfr/m']>=np.log10(mea)] #low
                hi=dd[dd['sfr/m']<np.log10(mea)] #high
    
            if splitMethod == 'prop':
                mea = np.median(dd[splitProp])
                lo=dd[dd[splitProp]<mea] #low
                hi=dd[dd[splitProp]>=mea] #high
    
            lo['split']='low'
            hi['split']='high'
    
            # exact merge
            merge = vstack([lo, hi])
    
            gen = Table.read(f'boxed_script/{name}/{name}_boxes.fits')
            merged_table = vstack([gen, merge])
            
            merged_table.write(f'boxed_script/{name}/{name}_boxes.fits', overwrite=True)
            
            # even split merge
            gen1 = Table.read(f'boxed_script/{name}/{name}_boxes_evenSplit.fits')
            
            low, high = match_table_lengths(lo,hi)
            
            evenSplit_merge = vstack([low, high])
            evenSplit = merged_table = vstack([gen1, evenSplit_merge])
            
            evenSplit.write(f'boxed_script/{name}/{name}_boxes_evenSplit.fits', overwrite=True)

    if preboxed == True & presplit==True:
        evenSplit=table
        evenSplit.write(f'boxed_script/{name}/{name}_boxes_evenSplit.fits')

    print('BOXING DONE')

    '''
    Grab low and high bins
    '''

    low = evenSplit[evenSplit['split']=='low']
    high = evenSplit[evenSplit['split']=='high']
    
    '''
    Jackknife and Cov
    '''
    if weights == 'uniform':
        m1,m2,c11,c22,c12=jackknife_weighted_mean_cov_fast2(low[f'profile_{filterType}_{frequency}'], np.ones(len(low)), 
                                                           high[f'profile_{filterType}_{frequency}'], np.ones(len(high)))

    if weights == 'mass':
        m1,m2,c11,c22,c12=jackknife_weighted_mean_cov_fast2(low[f'profile_{filterType}_{frequency}'], low['mass_weights'], 
                                                           high[f'profile_{filterType}_{frequency}'], high['mass_weights'])

    '''
    Plot
    '''
    r=np.array([2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. ])

    if splitMethod == 'prop':
        plt.errorbar(r, m1, yerr=np.diag(c11)**0.5, 
                     label=f"{'{:.2e}'.format(min(low[splitProp]))} < {splitProp} < {'{:.2e}'.format(max(low[splitProp]))} | low")
        plt.errorbar(r, m2, yerr=np.diag(c22)**0.5, 
                     label=f"{'{:.2e}'.format(min(high[splitProp]))} < {splitProp} < {'{:.2e}'.format(max(high[splitProp]))} | high")
        
    else:
        plt.errorbar(r, m1, yerr=np.diag(c11)**0.5, label='low feedback')
        plt.errorbar(r, m2, yerr=np.diag(c22)**0.5, label='high feedback')
    
    plt.axhline(y=0, color='black', linewidth=1, linestyle='--')
    plt.xlabel(r'$R$ [arcmin]')
    plt.ylabel(r'$T$ [$\mu K\cdot\mathrm{arcmin}^2$]')
    plt.title(f'Profile')
    plt.legend()
    
    plt.savefig(f'boxed_script/{name}/{name}_profile.png', bbox_inches='tight')
    
    #plt.show()
    plt.close()

    '''
    Save [radius, mean, and error] files
    '''
    np.savetxt(f'boxed_script/{name}/{name}_profile_lowf.txt', np.column_stack((r, m1, np.diag(c11)**0.5)), delimiter="\t")
    np.savetxt(f'boxed_script/{name}/{name}_profile_highf.txt', np.column_stack((r, m2, np.diag(c22)**0.5)), delimiter="\t")

    '''
    Save Cov and Chi2
    '''
    c_m = np.block([
        [c11, c12],
        [c12.T, c22]
    ])
    np.savetxt(f'boxed_script/{name}/{name}_cov.txt', c_m)
    
    c_d = c11+c22-c12- c12.T
    np.savetxt(f'boxed_script/{name}/{name}_crossCovariance.txt', c_d)

    chi=np.dot(np.dot((m1-m2),np.linalg.inv(c_d)), (m1-m2))
    np.savetxt(f'boxed_script/{name}/{name}_chi2.txt', np.array([chi]))

    return c_d, chi


def high_low_split(table, split_property, fixed_properties=['Z','LOGM'], n_boxes=100):
    """
    Bin objects into boxes by their fixed properties, then label each as
    high or low relative to its box's median in the split property.

    Objects are clustered into ``n_boxes`` groups via k-means on the
    ``fixed_properties`` (e.g. redshift and stellar mass), so that each box
    contains objects with similar values of those properties. Within each
    box, objects are split at the median of ``split_property``: those at or
    above the median are labeled ``high_<split_property>`` and those below
    are labeled ``low_<split_property>``. This controls for the fixed
    properties when comparing high- and low-split populations.

    Parameters
    ----------
    table : Table
        Input catalog containing ``split_property`` and all
        ``fixed_properties`` as columns. Not modified; a copy is returned.
    split_property : str
        Column name on which the high/low median split is performed.
    fixed_properties : list of str, optional
        Column names used as features for k-means clustering into boxes.
        Default ``['Z', 'LOGM']``.
    n_boxes : int, optional
        Number of k-means clusters (boxes) to form. Default 100.

    Returns
    -------
    Table
        A copy of the input table with two added columns:
        ``'box'`` (int), the cluster index for each object, and
        ``'split'`` (str), either ``high_<split_property>`` or
        ``low_<split_property>``.

    Notes
    -----
    Objects with ``split_property`` exactly equal to the box median are
    assigned to the high group. The ``fixed_properties`` are passed to
    k-means on their raw scales; consider standardizing them beforehand if
    they span very different ranges.
    """
    table = table.copy()
    data = np.vstack([table[p] for p in fixed_properties]).T
    centroids, _ = kmeans(data, n_boxes)
    labels, _ = vq(data, centroids)
    table['box'] = labels
    table['split'] = f'high_{split_property}'
    for i in range(n_boxes):
        in_box = (table['box'] == i)
        med = np.median(table[split_property][in_box])
        is_low = in_box & (table[split_property] < med)
        table['split'][is_low] = f'low_{split_property}'
    return table