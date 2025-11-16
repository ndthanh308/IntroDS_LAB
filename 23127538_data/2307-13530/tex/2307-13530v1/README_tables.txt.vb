\normalsize
\begin{verbatim}
The SOM/tables/ directory contains tables of occultation predictions in both typeset 
form and in machine-readable form, with with a sub-directory for each target:
        Jupiter, Saturn, Uranus, Neptune, Titan, and Triton. 

The table contents differ slightly for ringed/non-ringed planets and for the satellite targets. 

For example, the Neptune directory contains the following files:

./Neptune:
Neptune_tables.pdf               Complete set of typeset tables, produced
                                 from the following LaTeX source files:

Neptune_tables.tex               LaTeX file to produce Neptune_tables.pdf
Neptune_predictions_01_of_01.tex Data tables referred to in Neptune_tables.tex
aastex631.cls                    LaTeX style sheet used for Neptune_tables.tex

Note that to get proper column alignment in the typeset output file, the
input Neptune_tables.tex file must be typeset three times in succession.

Neptune_predictions_MR.txt       Machine-readable version of complete
                                 set of predictions for Neptune.
The format of the machine-readable file is listed at the beginning of the file. 

The header, file format, and first line of data (line-wrapped) for the file
Neptune/Neptune_predictions_MR.txt are as follows:

head -59 Neptune/Neptune_predictions_MR.txt | fold -124
\end{verbatim}
\scriptsize
\begin{verbatim}
Title: Earth-based Stellar Occultation Predictions for Jupiter, Saturn, Uranus, Neptune, Titan, and Triton: 2023-2050
Authors: Richard G. French &  Damya Souami
Table: Occultation predictions for Neptune 2023-2050 for K<15
============================================================
Byte-by-byte Description of file: Neptune_predictions_MR.txt
---------------------------------------------------------
   Bytes      Format    Units    Label     Explanations
---------------------------------------------------------
     1-  8       A8       ---               TARGET Occultation target
    10- 80      A71       ---          SUMMARY_PDF SOM pathname to event summary PDF
    82- 92      A11       ---              EVENTID Event identification (target letter, YYnnn)
    94-112    F19.7         d                   JD Julian date of closest approach (CA)
   114-140      A27       ---               UTC_CA UTC of closest approach
   142-151      A10       ---            EVENTTYPE Event type P: planet, g: geocentric t: topocentric
   153-174      I22       ---               STARID Gaia star ID
   176-211      A36       ---              STARPOS J2000 star position at epoch hh mm ss dd mm ss
   213-219     F7.1        km            STARERR_F E-W error in star position in skyplane (E positive)
   221-227     F7.1        km            STARERR_G N-S error in star position (N positive)
   229-264      A36       ---              TARGPOS J2000 target position at epoch hh mm ss dd mm ss
   266-274     F9.3    arcsec                   CA closest approach distance of target and star in skyplane
   276-285    F10.2       deg                   PA position angle of CA
   287-295     F9.3   1000 km               RKM_CA sky plane separation at CA
   297-305     F9.2      km/s                 VSKY sky plane velocity of target relative to star
   307-316    F10.3        au               DISTAU Observer-target distance
   318-327    F10.3      km/s                 RDOT ring plane radial velocity
   329-338    F10.2       deg                    P position angle of target pole
   340-349    F10.2       deg                 BDEG ring opening angle
   351-357     F7.1       deg                 LATI Ingress geodetic latitude
   359-367     F9.1       deg                 LATE Egress geodetic latitude
   369-377     F9.3       mag                 KMAG apparent K magnitude of occultation star
   379-388    F10.3       mag                 GMAG apparent G magnitude of occultation star
   390-399    F10.3       mag                GSTAR apparent G magnitude of occultation star, corrected for vsky
   401-410    F10.3       mag                RPMAG apparent RP magnitude of occultation star
   412-412       I1       ---                  DUP Source with multiple source identifiers (Gaia catalog entry)
   414-422     F9.3        km                SDIAM projected diameter of occultation star at target
   424-432     F9.1       deg               LONDEG Sub-target Earth longitude (East) at CA
   434-441     F8.1       deg               LATDEG Sub-target Earth latitude at CA
   443-450     F8.1       deg                  SGT Sun-Earth-Target separation at CA
   452-460     F9.1       deg                  MGT Moon-Earth-Target separation at CA
   462-469     F8.2       ---                 RUWE Renormalized unit weight error (<1.4 is good astronometric solution)
   471-491      A21       ---            _2MASS_ID 2MASS catalog ID
   493-493       I1       ---       _2MASS_DUPFLAG another nearby event with a different STARID shares this 2MASS catalog ID
   495-503     F9.3    arcsec                   _R angular separation of Gaia and 2MASS positions
   505-505       I1       ---     EAS_N_TARGETOCCS number of target occultations by candidate observatories in EAS region
   507-507       I1       ---       EAS_N_RINGOCCS number of ring occultations by candidate observatories in EAS region
   509-509       I1       ---     ENA_N_TARGETOCCS number of target occultations by candidate observatories in ENA region
   511-511       I1       ---       ENA_N_RINGOCCS number of ring occultations by candidate observatories in ENA region
   513-513       I1       ---     GEO_N_TARGETOCCS number of target occultations by candidate observatories in GEO region
   515-515       I1       ---       GEO_N_RINGOCCS number of ring occultations by candidate observatories in GEO region
   517-517       I1       ---     NAM_N_TARGETOCCS number of target occultations by candidate observatories in NAM region
   519-519       I1       ---       NAM_N_RINGOCCS number of ring occultations by candidate observatories in NAM region
   521-521       I1       ---     OCN_N_TARGETOCCS number of target occultations by candidate observatories in OCN region
   523-523       I1       ---       OCN_N_RINGOCCS number of ring occultations by candidate observatories in OCN region
   525-525       I1       ---     SAF_N_TARGETOCCS number of target occultations by candidate observatories in SAF region
   527-527       I1       ---       SAF_N_RINGOCCS number of ring occultations by candidate observatories in SAF region
   529-529       I1       ---     SAM_N_TARGETOCCS number of target occultations by candidate observatories in SAM region
   531-531       I1       ---       SAM_N_RINGOCCS number of ring occultations by candidate observatories in SAM region
---------------------------------------------------------
 Neptune   SOM/events/Neptune/2023/Neptune_2023-05-03T13_00_52.170_20230528a.pdf      N23001     2460068.0422705      2023-0
5-03 13:00:52.17      PgRgt    2447704782568325504      23 48 59.15652  -02 29 12.98136     4.4     3.0      23 48 59.14834 
 -02 29 12.68060     0.324     337.79     7.199     26.29     30.600     33.510     -41.58     -21.13     2.4     -34.4     
9.623     11.116     11.413     10.610 0     1.181     301.2     -2.4     46.1     160.6     0.96      23485925-0229122 0   
  0.324 0 0 0 0 1 1 0 1 0 0 0 0 0 0
\end{verbatim}
\normalsize
