"""Build a simulation-only K1/HX5 assembly without modifying source assets."""
from pathlib import Path
import xml.etree.ElementTree as ET
import copy,json,hashlib,shutil,subprocess,os
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def main():
    original=ROOT/'assets/source/k1_local.urdf';assets=ROOT/'assets/k1';assets.mkdir(exist_ok=True)
    meshdir=assets/'meshes'
    urdf=ET.parse(original).getroot()
    ET.SubElement(ET.SubElement(urdf,'mujoco'),'compiler',discardvisual='false',fusestatic='false',balanceinertia='false',strippath='false')
    for mesh in urdf.iter('mesh'):mesh.set('filename',str(meshdir/Path(mesh.get('filename')).name))
    # Replace the rubber hand link's geometry and inertia. Keep its wrist roll joint.
    for side in ['left','right']:
        link=urdf.find(f"link[@name='{side}_wrist_roll_rubber_hand']")
        for child in list(link):link.remove(child)
        # Temporary nonphysical placeholder only for URDF import; replaced before final compile.
        inertia=ET.SubElement(link,'inertial');ET.SubElement(inertia,'mass',value='.001');ET.SubElement(inertia,'inertia',ixx='1e-6',iyy='1e-6',izz='1e-6',ixy='0',ixz='0',iyz='0')
    local=assets/'k1_import.urdf';ET.ElementTree(urdf).write(local,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(local));mujoco.mj_saveLastXML(str(assets/'k1_import.xml'),model)
    robot=ET.parse(assets/'k1_import.xml').getroot();robot.set('model','k1_hx5_sim_assembly')
    compiler=robot.find('compiler');compiler.set('angle','radian');compiler.set('autolimits','true')
    # Saved URDF mesh paths may be basenames: resolve to the copied source mesh.
    for mesh in robot.findall('asset/mesh'):mesh.set('file',str(meshdir/Path(mesh.get('file')).name))
    option=robot.find('option')
    if option is None:option=ET.SubElement(robot,'option')
    option.attrib.update(timestep='.002',integrator='implicitfast',gravity='0 0 -9.81',cone='elliptic')
    pelvis=robot.find("worldbody/body[@name='pelvis']");assert pelvis is not None
    ET.SubElement(pelvis,'freejoint',name='root')
    defaults=robot.find('default')
    if defaults is None:defaults=ET.SubElement(robot,'default')
    actuator=robot.find('actuator')
    if actuator is None:actuator=ET.SubElement(robot,'actuator')
    contact=robot.find('contact')
    if contact is None:contact=ET.SubElement(robot,'contact')
    handroot=ROOT/'third_party/robotis_hand/robotis_hand_description'
    attachments=[]
    for side in ['left','right']:
        src=handroot/f'mujoco/hx5_d20_rev2/hx5_d20_{side}.xml';hand=ET.parse(src).getroot()
        for node in hand.find('default'):defaults.append(copy.deepcopy(node))
        for mesh in hand.findall('asset/mesh'):
            mesh=copy.deepcopy(mesh);mesh.set('file',str((src.parent/hand.find('compiler').get('meshdir')/mesh.get('file')).resolve()));robot.find('asset').append(mesh)
        wrist=next(b for b in robot.iter('body') if b.get('name')==f'{side}_wrist_roll_rubber_hand')
        for ch in list(wrist):
            if ch.tag!='joint':wrist.remove(ch)
        # Simulated mount: HX local +Z extends along K1 wrist +X. No hardware bracket claim.
        base=copy.deepcopy(hand.find('worldbody/body'));base.set('pos','0 0 0');base.set('quat','0.7071067811865476 0 0.7071067811865476 0')
        # Put the original root at the existing wrist joint, replacing the rubber hand.
        # Removing the empty wrapper lets the HX base inertia belong to wrist roll.
        wrist.set('childclass',base.get('childclass'))
        # Rotate the hand subtree inside a fixed mount; tiny wrapper has no inertia/geom.
        wrist.set('quat',base.get('quat'))
        wrist.find('joint').set('axis','0 0 1') # R_y(90) maps local +Z to original wrist +X
        for ch in list(base):wrist.append(ch)
        for ch in hand.find('actuator'):actuator.append(copy.deepcopy(ch))
        if hand.find('contact') is not None:
            for ch in hand.find('contact'):
                ch=copy.deepcopy(ch)
                for key in ['body1','body2']:
                    if ch.get(key)==base.get('name'):ch.set(key,wrist.get('name'))
                contact.append(ch)
        attachments.append({'side':side,'source_xml':str(src.relative_to(ROOT)),'source_sha256':sha(src),'parent':wrist.get('name'),'translation_m':[0,0,0],'quaternion_wxyz':[2**-.5,0,2**-.5,0],'status':'simulation_mount_assumption_not_hardware_verified'})
    # K1 torque motors. Effort limits come from the original URDF.
    bodyjoints=[j for j in urdf.findall('joint') if j.get('type')!='fixed'];body_names=[j.get('name') for j in bodyjoints]
    params={}
    for j in bodyjoints:
        name=j.get('name');limit=j.find('limit');cap=float(limit.get('effort'));speed=float(limit.get('velocity'))
        kp=180 if 'knee' in name else 140 if 'hip' in name else 100 if 'ankle' in name else 65 if 'waist' in name else 35
        kd=6 if 'knee' in name else 5 if 'hip' in name else 4 if 'ankle' in name else 3 if 'waist' in name else 2
        ET.SubElement(actuator,'motor',name=name+'_motor',joint=name,gear='1',ctrllimited='true',ctrlrange=f'{-cap} {cap}')
        params[name]={'kp':kp,'kd':kd,'effort':cap,'speed':speed}
    # Joint armature from the existing K1 actuator profile; no claim of motor identification.
    for j in robot.iter('joint'):
        if j.get('name') in body_names:
            big=('hip' in j.get('name') or 'knee' in j.get('name'))
            j.set('armature',str(.01936542 if big else .00564892));j.set('damping','0');j.set('frictionloss','0')
    world=robot.find('worldbody');ET.SubElement(world,'geom',name='floor',type='plane',size='10 10 .1',friction='.8 .005 .0001',rgba='.2 .23 .26 1')
    ET.SubElement(world,'light',name='key',pos='0 -2 4',dir='0 0 -1',diffuse='.8 .8 .8')
    # Exclude adjacent links as usual, retain all other self collisions.
    for body in robot.iter('body'):
        for child in body.findall('body'):ET.SubElement(contact,'exclude',body1=body.get('name'),body2=child.get('name'))
    path=ROOT/'assets/k1_hx5.xml';ET.indent(robot);
    for mesh in robot.findall('asset/mesh'):mesh.set('file',os.path.relpath(mesh.get('file'),path.parent))
    ET.ElementTree(robot).write(path,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    assert model.nq==70 and model.nv==69 and model.nu==63,(model.nq,model.nv,model.nu)
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,model.njnt)]
    neutral=np.zeros(model.nq);neutral[3]=1
    for n in body_names:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n);v=-.18 if 'hip_pitch' in n or 'ankle_pitch' in n else .36 if 'knee' in n else .95 if 'elbow' in n else .15 if 'shoulder_pitch' in n else .15 if n=='left_shoulder_roll_joint' else -.15 if n=='right_shoulder_roll_joint' else 0
        neutral[model.jnt_qposadr[jid]]=v
    # Finger zero pose, clipped inside official ranges, is a fixed motor target.
    for i in range(1,model.njnt):neutral[model.jnt_qposadr[i]]=np.clip(neutral[model.jnt_qposadr[i]],*model.jnt_range[i])
    data.qpos[:]=neutral;mujoco.mj_forward(model,data)
    # Ground height from actual official foot collision spheres, not visual meshes.
    foot=[g for g in range(model.ngeom) if model.geom_contype[g] and 'ankle' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g]) or '')]
    bottom=min(data.geom_xpos[g,2]-model.geom_size[g,0] for g in foot if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE)
    neutral[2]=-bottom+.001;data.qpos[:]=neutral;mujoco.mj_forward(model,data)
    np.save(ROOT/'assets/neutral_qpos.npy',neutral)
    manifest={'status':'model_compiled_not_dynamic_tracking_validation','nq':model.nq,'nv':model.nv,'nu':model.nu,'body_joint_names':body_names,'hand_joint_names':[n for n in names if n not in body_names],'total_mass_kg':float(model.body_mass.sum()),'original_K1_mass_kg':sum(float(x.get('value')) for x in ET.parse(original).getroot().findall('link/inertial/mass')),'rubber_hand_mass_removed_kg':.738,'attachments':attachments,'source_K1_urdf_sha256':sha(original),'hand_revision':json.loads((ROOT/'target.json').read_text())['pinned_commits']['HX5'],'model_sha256':sha(path),'body_actuator_parameters':params,'neutral_root_height_m':float(neutral[2]),'hand_controller':'official_position_actuators_at_zero_clipped_to_range','hand_manipulation_in_scope':False}
    write(ROOT/'reports/robot_manifest.json',manifest);write(ROOT/'configs/assembly.json',{'attachments':attachments,'physics_dt':.002,'control_dt':.02,'hand_target':'zero_clipped_to_official_joint_ranges'})
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['attachments','body_actuator_parameters','body_joint_names','hand_joint_names']},indent=2))
if __name__=='__main__':main()
