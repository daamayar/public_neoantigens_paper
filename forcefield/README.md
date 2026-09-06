# Force field files

The simulations use CHARMM36m with the TIP3P water model. Most of the parameter
files carry their own licence terms and are therefore not redistributed here.
Place them in this directory, or point `CHARMM_FF_DIR` at wherever you keep them.

## Included

| File | Origin |
|---|---|
| `par_water_ions.prm` | Derived here. See below. |

The NAMD-formatted `toppar_water_ions_namd.str` has its `read para` directives
commented out, so ParmEd cannot parse its parameter section and the water and ion
terms are silently dropped. The bond, angle, nonbonded and NBFIX parameters were
therefore extracted into `par_water_ions.prm`, which ParmEd reads normally. Both
files are loaded: the `.str` supplies the RTF section, the `.prm` the parameters.

## To be supplied

From the CHARMM36m distribution (MacKerell lab,
<https://www.charmm.org/charmm/resources/charmm-force-fields/>, or the `toppar`
directory of a NAMD installation):

```
top_all36_prot.rtf
par_all36m_prot.prm
par_all36_lipid.prm
par_all36_carb.prm
par_all36_cgenff_namd.prm
toppar_water_ions_namd.str
```

From any VMD installation (the solvate plugin):

```
wat.top          lib/vmd/plugins/noarch/tcl/solvate<version>/wat.top
```

`wat.top` defines the TIP3P residue and its atom types (`OT`, `HT`), which the
CHARMM protein RTF does not provide. Without it ParmEd cannot type the solvent
and topology conversion fails. Point `VMD_WAT_TOP` at the file if you prefer not
to copy it here.

## Checking

```bash
cd amber_free_energy
python -c "from scripts import config; config.validate_force_field()"
```

The call is silent when every file is present and otherwise lists exactly what is
missing and where it was looked for.
