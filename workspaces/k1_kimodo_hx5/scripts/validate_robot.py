"""Independent original-K1 vs assembled-K1 FK and mass checks."""
import xml.etree.ElementTree as E
from pathlib import Path
import json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
def main():
    original=ROOT/'assets/source/k1_local.urdf';r=E.parse(original).getroot();E.SubElement(E.SubElement(r,'mujoco'),'compiler',strippath='false',fusestatic='false')
    for mesh in r.iter('mesh'):mesh.set('filename',str(ROOT/'assets/k1/meshes'/Path(mesh.get('filename')).name))
    urdf=ROOT/'assets/original_K1_fixture.urdf';E.ElementTree(r).write(urdf,encoding='unicode');om=mujoco.MjModel.from_xml_path(str(urdf));mujoco.mj_saveLastXML(str(ROOT/'assets/original_K1_fixture.xml'),om)
    omr=E.parse(ROOT/'assets/original_K1_fixture.xml').getroot();E.SubElement(omr.find('worldbody/body'),'freejoint',name='fixture_root');om=mujoco.MjModel.from_xml_string(E.tostring(omr,encoding='unicode'));od=mujoco.MjData(om)
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/k1_hx5.xml'));d=mujoco.MjData(m);manifest=json.loads((ROOT/'reports/robot_manifest.json').read_text());ns=manifest['body_joint_names'];neutral=np.load(ROOT/'assets/neutral_qpos.npy');rng=np.random.default_rng(5721);maxpos=0;maxrot=0
    originalnames=[mujoco.mj_id2name(om,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,om.nbody)]
    for case in range(11):
        d.qpos[:]=neutral;od.qpos[:7]=neutral[:7]
        quat=Rotation.from_euler('xyz',rng.uniform(-.2,.2,3)).as_quat();d.qpos[3:7]=od.qpos[3:7]=quat[[3,0,1,2]]
        for name in ns:
            j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);oj=mujoco.mj_name2id(om,mujoco.mjtObj.mjOBJ_JOINT,name)
            q=np.clip(neutral[m.jnt_qposadr[j]]+rng.uniform(-.2,.2),*m.jnt_range[j]);d.qpos[m.jnt_qposadr[j]]=od.qpos[om.jnt_qposadr[oj]]=q
        mujoco.mj_forward(m,d);mujoco.mj_forward(om,od)
        for name in originalnames:
            i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name);oi=mujoco.mj_name2id(om,mujoco.mjtObj.mjOBJ_BODY,name)
            maxpos=max(maxpos,float(np.linalg.norm(d.xpos[i]-od.xpos[oi])))
            expected=od.xmat[oi].reshape(3,3)
            if 'rubber_hand' in name:expected=expected@Rotation.from_euler('y',90,degrees=True).as_matrix()
            maxrot=max(maxrot,float(np.linalg.norm(Rotation.from_matrix(expected.T@d.xmat[i].reshape(3,3)).as_rotvec())))
    handmass=0
    for side in ['left','right']:
        hr=E.parse(ROOT/f'third_party/robotis_hand/robotis_hand_description/mujoco/hx5_d20_rev2/hx5_d20_{side}.xml').getroot()
        handmass+=sum(float(x.get('mass')) for x in hr.findall('.//inertial'))
    expectedmass=om.body_mass.sum()-.738+handmass
    checks={'63_actuators':m.nu==63,'23_body_plus_40_hand_joints':len(manifest['body_joint_names'])==23 and len(manifest['hand_joint_names'])==40,'source_K1_FK_position_under_1um':maxpos<1e-6,'wrist_mount_rotation_and_roll_axis_under_1urad':maxrot<1e-6,'mass_no_double_count':bool(abs(m.body_mass.sum()-expectedmass)<1e-7),'positive_inertias':bool(np.all(m.body_inertia[1:]>0))}
    report={'scope':'11_asymmetric_pose_fixtures_against_original_K1_model_not_hardware_mount_certification','checks':checks,'passed':all(checks.values()),'max_position_error_m':maxpos,'max_rotation_error_rad':maxrot,'original_mass_kg':float(om.body_mass.sum()),'official_hand_pair_mass_kg':handmass,'combined_expected_mass_kg':float(expectedmass),'combined_actual_mass_kg':float(m.body_mass.sum())}
    (ROOT/'reports/robot_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
