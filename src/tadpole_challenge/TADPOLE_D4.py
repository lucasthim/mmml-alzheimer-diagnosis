'''
Assemble the D4 spreadsheet from the Freesurfer output and ADNIMERGE.

Author: Razvan V. Marinescu
'''


import pandas as pd
import os
import numpy as np
import pickle
import matplotlib
from matplotlib import pyplot as pl
from sklearn import datasets, linear_model


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 55)
pd.set_option('display.width', 5000)

def agregateFSData(fsSubjDir):


  exportSubjCmd = 'export SUBJECTS_DIR=%s' % fsSubjDir
  exportFreeSurfCmd = 'export FREESURFER_HOME=%s; source %s/SetUpFreeSurfer.sh' \
    % (freesurfPath, freesurfPath)

  from glob import glob
  subjFlds = [x.split('/')[-2] for x in glob("%s/*/" % fsSubjDir)]


  subjListStr = ' '.join(subjFlds)
  print(subjListStr)
  runAggreg = True

  subcortProg = 'asegstats2table'
  subcortOutFile = '%s/fsxSubcort.csv' % fsSubjDir
  subcortFsAggCmd = '%s ; %s; %s/bin/%s --subjects %s --meas volume --skip ' \
    '--tablefile %s --delimiter=comma ' % (exportSubjCmd, exportFreeSurfCmd,
    freesurfPath, subcortProg, subjListStr, subcortOutFile)
  print(subcortFsAggCmd)
  if runAggreg:
    os.system(subcortFsAggCmd)

  dfSubcort = pd.read_csv(subcortOutFile)

  cortProg = 'aparcstats2table'
  cortLhOutFile = '%s/fsxCortLh.csv' % fsSubjDir
  cortLhFsAggCmd = '%s ; %s; %s/bin/%s --subjects %s --meas volume --hemi lh --skip ' \
                    '--tablefile %s --delimiter=comma ' % (exportSubjCmd, exportFreeSurfCmd,
  freesurfPath, cortProg, subjListStr, cortLhOutFile)
  print(cortLhFsAggCmd)
  if runAggreg:
    os.system(cortLhFsAggCmd)
  dfCortLh = pd.read_csv(cortLhOutFile)

  cortRhOutFile = '%s/fsxCortRh.csv' % fsSubjDir
  cortRhFsAggCmd = '%s ; %s; %s/bin/%s --subjects %s --meas volume --hemi rh --skip ' \
                   '--tablefile %s --delimiter=comma ' % (exportSubjCmd, exportFreeSurfCmd,
  freesurfPath, cortProg, subjListStr, cortRhOutFile)
  print(cortRhFsAggCmd)
  if runAggreg:
    os.system(cortRhFsAggCmd)
  dfCortRh = pd.read_csv(cortRhOutFile)

  assert dfSubcort.shape[0] == dfCortLh.shape[0] == dfCortRh.shape[0]

  dfMri = dfSubcort
  dfMri[dfCortLh.columns] = dfCortLh
  dfMri[dfCortRh.columns] = dfCortRh

  # print(adss)

  dfMri['RID'] = [int(x.split('-')[0][-4:]) for x in dfMri['Measure:volume']]
  dfMri['studyID'] = [int(x.split('-')[0][-4:]) for x in dfMri['Measure:volume']]
  dfMri['ScanDate'] = [x[11:21] for x in dfMri['Measure:volume']]

  dfMri.to_csv('dfMri.csv', index=False)

  return dfMri

def calcVentVol(mriAll):
  allCols = list(mriAll.columns)
  ventCols = [c for c in allCols if (c.split('_')[0] in ['Left-Inf-Lat-Vent', 'Left-Lateral-Ventricle',
                                                         'Right-Inf-Lat-Vent', 'Right-Lateral-Ventricle'])]
  print('ventCols', ventCols)
  # ventSum = np.sum(d12Df.loc[:,ventCols], axis=1)
  ventSum = np.sum((mriAll[ventCols]).apply(pd.to_numeric, errors='coerce'), axis=1)

  return ventSum

def makeD4Dataset(mriAll, adniMerge, d12Df, ventsD4Corr):

  d2Rids = np.unique(d12Df.RID[d12Df.D2 == 1])

  print('----------')
  print(np.sum(adniMerge.ORIGPROT == 'ADNI3'), adniMerge.loc[adniMerge.ORIGPROT == 'ADNI3', : 'EXAMDATE'])
  print(np.sum((adniMerge.EXAMDATE >= '2018-01-01')), adniMerge.loc[adniMerge.EXAMDATE >= '2018-01-01', : 'EXAMDATE'])
  print(np.sum((np.in1d(adniMerge.RID, d2Rids))), adniMerge.loc[np.in1d(adniMerge.RID, d2Rids), : 'EXAMDATE'])
  adni3And2018Mask = np.logical_and((adniMerge.COLPROT == 'ADNI3'),
    (adniMerge.EXAMDATE >= '2018-01-01'))
  d4Indx = np.logical_and(adni3And2018Mask, np.in1d(adniMerge.RID, d2Rids))
  adniMerge['EXAMDATE'] = pd.to_datetime(adniMerge['EXAMDATE'])
  adniMergeD4 = adniMerge[d4Indx]
  adniMergeD4.reset_index(drop=True, inplace=True)

  d4Df = pd.DataFrame(np.nan,index=range(adniMergeD4.shape[0]), columns=('RID', 'LB4', 'CognitiveAssessmentDate', 'Diagnosis', 'ADAS13', 'ScanDate', 'Ventricles'))


  d4Df['RID'] = adniMergeD4['RID']
  d4Df['LB4'] = np.ones(adniMergeD4.shape[0])
  d4Df['CognitiveAssessmentDate'] = adniMergeD4['EXAMDATE']
  d4Df['Diagnosis'] = adniMergeD4['DX']
  d4Df['ADAS13'] = adniMergeD4['ADAS13']
  d4Df['ScanDate'] = d4Df['CognitiveAssessmentDate'] # fill up ScanDate even if there are no values.

  d4Df['DX_bl'] = adniMergeD4['DX_bl']
  d4Df['DX_LastVisitADNI2'] = ' '
  d4Df['AGE'] = adniMergeD4['AGE'] + adniMergeD4['Years_bl']
  d4Df['PTGENDER'] = adniMergeD4['PTGENDER']
  d4Df['MMSE'] = adniMergeD4['MMSE']
  d4Df['Years_bl'] = adniMergeD4['Years_bl']

  mapping = {'Dementia':'AD'}
  mask = np.logical_not(np.in1d(d4Df.Diagnosis, ['CN', 'MCI']))
  d4Df.loc[mask, 'Diagnosis'] = d4Df.loc[mask, 'Diagnosis'].map(mapping)

  ### the ventricles and ICV are calculated as per the ADNIMERGER R package by Mike Donohue.
  ## Raz: I used the TADPOLE_D1_D2_Dict.csv to decode the STxxSV labels to Left-Inf-Lat-Ventricle, etc ...
  # 'ST30SV', 'ST37SV', 'ST89SV', 'ST96SV' -> 'Left-Inf-Lat-Vent', 'Left-Lateral-Ventricle',
  #   'Right-Inf-Lat-Vent', 'Right-Lateral-Ventricle'
  # For ICV Mike used ST10CV -> i.e. IntraCranialVol

  ventNorm = ventsD4Corr

  d4Df.CognitiveAssessmentDate = pd.to_datetime(d4Df['CognitiveAssessmentDate'], format='%Y-%m-%d')
  mriAll.ScanDate = pd.to_datetime(mriAll.ScanDate, format='%Y-%m-%d')
  d4Df.ScanDate = pd.to_datetime(d4Df['ScanDate'], format='%Y-%m-%d')

  for s in range(mriAll.shape[0]):
    notMriMask = np.isnan(d4Df.Ventricles)
    indCurrSubj = np.where(np.logical_and(d4Df.RID == mriAll.RID[s], notMriMask))[0]
    assDatesCurr = d4Df.loc[indCurrSubj, 'CognitiveAssessmentDate']

    # now find a target spot to place this scan info.
    if assDatesCurr.shape[0] == 0:
      # no suitable entry was found, so simply create a new row

      lb4Mri = pd.DataFrame(index=range(1), columns=d4Df.columns)
      lb4Mri['RID'] = mriAll.RID[s]
      lb4Mri['ScanDate'] = mriAll.ScanDate[s]
      lb4Mri['Ventricles'] = ventNorm[s]
      lb4Mri['LB4'] = 1
      lb4Mri['CognitiveAssessmentDate'] = mriAll.ScanDate[s]  # fill up ScanDate even if there are no values.

      # pull other variables from the full ADNIMerge (D1/D2)
      idxSomeOlderEntryInAdni2 = np.where(adniMerge.RID == mriAll.RID[s])[0][0]
      lb4Mri['DX_bl'] = adniMerge.loc[idxSomeOlderEntryInAdni2, 'DX_bl']
      lb4Mri['AGE'] = float((mriAll.ScanDate[s] - adniMerge['EXAMDATE'][idxSomeOlderEntryInAdni2]).days)/365 + \
                      adniMerge['AGE'][idxSomeOlderEntryInAdni2]
      lb4Mri['PTGENDER'] = adniMerge.PTGENDER[idxSomeOlderEntryInAdni2]
      lb4Mri['Years_bl'] = float((mriAll.ScanDate[s] - adniMerge['EXAMDATE'][idxSomeOlderEntryInAdni2]).days)/365 + \
                           adniMerge['Years_bl'][idxSomeOlderEntryInAdni2]

      d4Df = d4Df.append(lb4Mri, ignore_index=True)

    else:
      indClosestDate = np.argmin(np.abs(np.array(assDatesCurr) - mriAll.ScanDate.values[s]))

      d4Df.loc[indCurrSubj[indClosestDate], 'ScanDate'] = mriAll.ScanDate[s]
      d4Df.loc[indCurrSubj[indClosestDate], 'Ventricles'] = ventNorm[s]

  # d4Df['ScanDate'] = adniMergeD4['EXAMDATE'] # for now set the scan date to be EXAMDATE
  # d4Df['Ventricles'] =


  unqRID = np.unique(d4Df.RID)
  for p in range(unqRID.shape[0]):
    indCurr = np.where(d4Df.RID == unqRID[p])[0]
    indOfFirstVisitInD4 = np.argmin(d4Df.Years_bl.values[indCurr])
    d4Df.loc[indCurr, 'Years_bl'] = d4Df.Years_bl[indCurr] - d4Df.Years_bl[indCurr[indOfFirstVisitInD4]]
    d4Df.loc[indCurr, 'DX_bl'] = d4Df.Diagnosis[indCurr[indOfFirstVisitInD4]]

    maskCurrInAdni2 = np.logical_and(adniMerge.RID == unqRID[p], ~pd.isnull(adniMerge.DX))
    maskCurrInAdni2 = np.logical_and(maskCurrInAdni2, adniMerge.EXAMDATE <= '2017-12-31')
    examDatesInAdni2 = adniMerge.EXAMDATE[maskCurrInAdni2]

    if examDatesInAdni2.shape[0] > 0:
      indLastVisitAdni2 = np.argmin(np.abs(examDatesInAdni2 - d4Df.CognitiveAssessmentDate[indCurr[indOfFirstVisitInD4]]))
      d4Df.loc[indCurr, 'DX_LastVisitADNI2'] = adniMerge.loc[indLastVisitAdni2, 'DX']

    if pd.isnull(d4Df.Diagnosis[indCurr[indOfFirstVisitInD4]]):
      d4Df.loc[indCurr, 'DX_bl'] = d4Df.loc[indCurr, 'DX_LastVisitADNI2']

  d4Df.DX_LastVisitADNI2 = d4Df.DX_LastVisitADNI2.map({'CN':'CN', 'MCI':'MCI', 'Dementia':'AD'})

  d4Df = d4Df.sort_values(by=['RID', 'CognitiveAssessmentDate', 'ScanDate'])
  entriesWithTargetVars = np.sum(pd.isnull(d4Df[['Diagnosis', 'ADAS13', 'Ventricles']]), axis=1) < 3
  d4Df = d4Df[entriesWithTargetVars]

  return d4Df


def biasCorrVents(mriAllD4, mriAllD12, d12Df, adniMerge):


  ventsOurs = calcVentVol(mriAllD12)

  mriAllD12.ScanDate = pd.to_datetime(mriAllD12['ScanDate'], format='%Y-%m-%d')
  d12Df.EXAMDATE = pd.to_datetime(d12Df['EXAMDATE'], format='%Y-%m-%d')

  mriAllD12['ventsFsOurs'] = ventsOurs
  mriAllD12['ventsFsTheirs'] = np.zeros(mriAllD12.shape[0], float)
  for s in range(mriAllD12.shape[0]):
    ridCurr = mriAllD12['RID'][s]
    scanDateCurr = mriAllD12['ScanDate'][s]

    idxSubjInD12 = np.where(d12Df.loc[:, 'RID'] == ridCurr)[0]
    closestEntry = np.argmin((np.abs(d12Df.loc[idxSubjInD12, 'EXAMDATE'] - scanDateCurr)).values)

    mriAllD12.loc[s, 'ventsFsTheirs'] = d12Df.loc[idxSubjInD12[closestEntry],'Ventricles']
    mriAllD12.loc[s, 'icvFsTheirs'] = d12Df.loc[idxSubjInD12[closestEntry], 'ICV']


  mriAllD12.loc[:, 'ventsFsOurs'] /= mriAllD12.loc[:, 'IntraCranialVol']
  mriAllD12.loc[:, 'ventsFsTheirs'] /= mriAllD12.loc[:, 'icvFsTheirs']

  # add significantly more bias, then show plots below, to see if the computation is correct
  # mriAllD12.loc[:, 'ventsFsOurs'] = 2 * mriAllD12.loc[:, 'ventsFsOurs'] + 0.05


  regr = linear_model.LinearRegression()
  nnMask = np.logical_and(np.isfinite(mriAllD12.loc[:, 'ventsFsOurs'].values), np.isfinite(mriAllD12.loc[:, 'ventsFsTheirs'].values))
  regr.fit(mriAllD12.loc[nnMask, 'ventsFsOurs'].values.reshape(-1,1), mriAllD12.loc[nnMask, 'ventsFsTheirs'].values.reshape(-1,1))

  mriAllD12['ventsFsCorr'] = regr.predict(mriAllD12.loc[:, 'ventsFsOurs'].values.reshape(-1,1))

  maeBefore = np.mean(np.abs(mriAllD12.loc[nnMask, 'ventsFsOurs'].values - mriAllD12.loc[nnMask, 'ventsFsTheirs'].values))
  maeAfter = np.mean(np.abs(mriAllD12.loc[nnMask, 'ventsFsCorr'].values - mriAllD12.loc[nnMask, 'ventsFsTheirs'].values))
  print('mae before ventricle correction', maeBefore)
  print('mae after ventricle correction', maeAfter)

  fig = pl.figure(1)

  pl.scatter(mriAllD12.loc[:, 'ventsFsOurs'], mriAllD12.loc[:, 'ventsFsTheirs'], c='r')
  pl.scatter(mriAllD12.loc[:, 'ventsFsCorr'], mriAllD12.loc[:, 'ventsFsTheirs'], c='g')
  fig.show()


  ventsD4Init = calcVentVol(mriAllD4)
  ventsD4Norm = ventsD4Init.values / mriAllD4['IntraCranialVol'].values  # fs 5.1.0
  ventsD4Corr = regr.predict(ventsD4Norm.reshape(-1,1))

  return ventsD4Corr

# first pull the data from the cluster using make pullStatsFS

# freesurfPath = '/home/rmarines/src/freesurfer-5.1.0'
freesurfPath = '/data/vision/polina/shared_software/freesurfer_v5.1.0'


fsSubjDir = 'fs-subjects-04-02-2019'
# mriAll = agregateFSData(fsSubjDir)
mriAll = pd.read_csv('dfMri.csv')
adniMerge = pd.read_csv('ADNIMERGE.csv')
d12Df = pd.read_csv('TADPOLE_D1_D2.csv')
os.system('mkdir generated')
pickle.dump(dict(mriAll=mriAll, adniMerge=adniMerge, d12Df=d12Df),
  open('generated/tempD4.npz', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

ds = pickle.load(open('generated/tempD4.npz', 'rb'))
mriAllD4 = ds['mriAll']
# mriAll.to_csv('D4_mri.csv')
adniMerge = ds['adniMerge']
d12Df = ds['d12Df']

mriAllD12 = pd.read_csv('dfMri_D12.csv')
ventsD4Corr = biasCorrVents(mriAllD4, mriAllD12, d12Df, adniMerge)

d4Df = makeD4Dataset(mriAllD4, adniMerge, d12Df, ventsD4Corr)
# lb4Df.to_csv('TADPOLE_D4.csv',index=False)
d4Df.to_csv('TADPOLE_D4_corr.csv', index=False) # 'corr' means ventricles are bias-corrected for FS version and OS architecture
