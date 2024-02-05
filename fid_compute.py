from cleanfid import fid

# fid_f = fid.compute_fid('/data/abhijnya/Classifier_guidance/runs/rebuttal_distribution/0,1,0,0/dist_flipped', '/data/abhijnya/dataset/celebahq/CelebA-HQ-img')
fid_u = fid.compute_fid('/data/abhijnya/Classifier_guidance/runs/rebuttal_distribution_more_flipped/0,1,0,0/dist_unflipped_2k', '/data/abhijnya/dataset/celebahq/CelebA-HQ-img')

# kid_f = fid.compute_kid('/data/abhijnya/Classifier_guidance/runs/rebuttal_distribution/0,1,0,0/dist_flipped', '/data/abhijnya/dataset/celebahq/CelebA-HQ-img')
kid_u = fid.compute_kid('/data/abhijnya/Classifier_guidance/runs/rebuttal_distribution_more_flipped/0,1,0,0/dist_unflipped_2k', '/data/abhijnya/dataset/celebahq/CelebA-HQ-img')

print("fid u, kid u")
print(fid_u, kid_u)