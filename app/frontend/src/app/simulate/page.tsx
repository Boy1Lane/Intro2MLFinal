import { redirect } from "next/navigation";

// Batch CSV scoring is now a mode inside the Studio dashboard (#analyze section).
export default function SimulatePage() {
  redirect("/#analyze");
}
