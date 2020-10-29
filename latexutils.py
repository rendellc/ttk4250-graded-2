import csv

PARAMETER_TO_TEXNAME = dict(
    # cont_gyro_noise_std = r"\sigma_{gyro}",
    # cont_acc_noise_std = r"\sigma_{acc}",
    # rate_bias_driving_noise_std = r"\sigma_{w,gyro}",
    # acc_bias_driving_noise_std = r"\sigma_{w,acc}",
    p_gyro = r"p_\text{gyro}",
    p_acc = r"p_\text{acc}",
    sigma_pos = r"\sigma_{p0}",
    sigma_vel = r"\sigma_{v0}",
    sigma_err_acc_bias = r"\sigma_{ab0}",
    sigma_err_gyro_bias = r"\sigma_{gb0}",
)

def sig_exp(num_str):
    parts = num_str.split('.', 2)
    decimal = parts[1] if len(parts) > 1 else ''
    exp = -len(decimal)
    digits = parts[0].lstrip('0') + decimal
    trimmed = digits.rstrip('0')
    exp += len(digits) - len(trimmed)
    sig = int(trimmed) if trimmed else 0
    return sig, exp

def sig_exp_e(num_str):
    v, exp = num_str.split('e')
    return float(v), int(exp)

def parameter_to_texvalues(params):
    parameter_tex_dict = {}
    for paramname,value in params.items():
        if paramname in PARAMETER_TO_TEXNAME.keys():
            tex_key = PARAMETER_TO_TEXNAME[paramname]
            parameter_tex_dict[tex_key] = value


    # need to set these manually
    p = params
    pt = PARAMETER_TO_TEXNAME
    sqdt = p["dt"]**0.5

    pa, pg = p["p_acc"], p["p_gyro"]

    pa_v, pa_exp = sig_exp_e(str(pa))
    pg_v, pg_exp = sig_exp_e(str(pg))
    pa_v_str = "" if abs(pa_v-1) < 0.0001 else str(pa_v)
    pg_v_str = "" if abs(pg_v-1) < 0.0001 else str(pg_v)
    parameter_tex_dict[pt["p_acc"]] = pa_v_str +" 10^{" + str(pa_exp) + "}"
    parameter_tex_dict[pt["p_gyro"]] = pg_v_str +" 10^{" + str(pg_exp) + "}"

    parameter_tex_dict[r"\sigma_\text{gyro}"] = f"{p['rate_std_factor']*p['cont_gyro_noise_std']* 1/sqdt:.7f}"
    parameter_tex_dict[r"\sigma_\text{acc}"] = f"{p['acc_std_factor']*p['cont_acc_noise_std']* 1/sqdt:.7f}"
    parameter_tex_dict[r"\sigma_\text{gyro,bias}"] = f"{p['rate_std_factor']*p['cont_gyro_noise_std']* 1/sqdt:.7f}"
    parameter_tex_dict[r"\sigma_\text{acc,bias}"] = f"{p['acc_std_factor']*p['cont_gyro_noise_std']* 1/sqdt:.7f}"

    p_std_2 = [f"{v}^2" for v in p['p_std']]
    parameter_tex_dict[r"\mathbf{R}_\text{GNSS}"] = r"\text{diag}" + rf"([{p_std_2[0]}, {p_std_2[1]}, {p_std_2[2]}])"

    return parameter_tex_dict


def save_params_to_csv(params, filename, headers=[]):
    #texparameters = parameter_to_texvalues(params)

    with open(filename, 'w', newline='') as csvfile:
        print("Writing parameters to", csvfile.name)
        writer = csv.writer(csvfile, delimiter=';',quoting=csv.QUOTE_MINIMAL)

        # write so that latex will interpret it correctly
        # cant start a line with backslash so first column is empty
        if headers:
            writer.writerow(["", *headers])
        for k,v in params.items():
            writer.writerow(["", k, v])

